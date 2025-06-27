#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, strip_symmetric, build_scaling_rotation, build_rotation, quaternion_multiply
from utils.debug_utils import save_cal_graph, save_tensor_img
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH, rotate_sh_by_matrix, rotate_sh_by_quaternion
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
import tinycudann as tcnn
from ntc import NeuralTransformationCache
import commentjson as ctjs

from sklearn.cluster import DBSCAN
from scipy.spatial import cKDTree
from PIL import Image
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt

class GaussianModel:

    def setup_functions(self):
        
        # @torch.compile
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
                
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation
        self.rotation_compose = quaternion_multiply
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, sh_degree : int, rotate_sh:bool = False):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        
        self._xyz_bound_min = None
        self._xyz_bound_max = None
        
        self._d_xyz = None
        self._d_rot = None
        self._d_rot_matrix = None
        self._d_scaling = None
        self._d_opacity = None
        
        self._new_xyz = None
        self._new_rot = None
        self._new_scaling = None
        self._new_opacity = None
        self._new_feature = None
        self._rotate_sh=rotate_sh
        
        self._added_xyz = None
        self._added_features_dc = None
        self._added_features_rest = None
        self._added_opacity = None
        self._added_scaling = None
        self._added_rotation = None
        self._added_mask = None
        
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.color_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        if self._new_scaling is not None:
            return self._new_scaling
        elif self._added_scaling is not None:
            return self.scaling_activation(torch.cat((self._scaling, self._added_scaling), dim=0))
        else:
            return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        if self._new_rot is not None:
            return self._new_rot
        elif self._added_rotation is not None:
            return self.rotation_activation(torch.cat((self._rotation, self._added_rotation), dim=0))
        else:
            return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        if self._new_xyz is not None:
            return self._new_xyz
        elif self._added_xyz is not None:
            return torch.cat((self._xyz, self._added_xyz), dim=0)
        else:
            return self._xyz
    
    @property
    def get_features(self):
        if self._new_feature is not None:
            return self._new_feature
        elif self._added_features_dc is not None and self._added_features_rest is not None:
            features_dc = torch.cat((self._features_dc, self._added_features_dc), dim=0)
            features_rest = torch.cat((self._features_rest, self._added_features_rest), dim=0)
            return torch.cat((features_dc, features_rest), dim=1)
        else:
            features_dc = self._features_dc
            features_rest = self._features_rest
            return torch.cat((features_dc, features_rest), dim=1)  
          
    @property
    def get_opacity(self):
        if self._new_opacity is not None:
            return self._new_opacity
        elif self._added_opacity is not None:
            return self.opacity_activation(torch.cat((self._opacity, self._added_opacity), dim=0))
        else:
            return self.opacity_activation(self._opacity)
        
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self.get_rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.color_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path, save_type='all'):
        mkdir_p(os.path.dirname(path))
        if save_type=='added':
            xyz = self._added_xyz.detach().cpu().numpy()
            normals = np.zeros_like(xyz)
            f_dc = self._added_features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            f_rest = self._added_features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            opacities = self._added_opacity.detach().cpu().numpy()
            scale = self._added_scaling.detach().cpu().numpy()
            rotation = self._added_rotation.detach().cpu().numpy()       
        elif save_type=='origin':  
            xyz = self._xyz.detach().cpu().numpy()
            normals = np.zeros_like(xyz)
            f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            opacities = self._opacity.detach().cpu().numpy()
            scale = self._scaling.detach().cpu().numpy()
            rotation = self._rotation.detach().cpu().numpy()
        elif save_type=='all':
            xyz = self.get_xyz.detach().cpu().numpy()
            normals = np.zeros_like(xyz)
            f_dc = self.get_features[:,0:1,:].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            f_rest = self.get_features[:,1:,:].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            opacities = self.inverse_opacity_activation(self.get_opacity).detach().cpu().numpy()
            scale = self.scaling_inverse_activation(self.get_scaling).detach().cpu().numpy()
            rotation = self.get_rotation.detach().cpu().numpy()
        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]  
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path, spatial_lr_scale=0):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self.spatial_lr_scale = spatial_lr_scale
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            #print(f"What is in the optimizer for {group['name']}: {group['params'][0].shape}, mask shape: {mask.shape}")
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.color_gradient_accum = self.color_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.color_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        prune_mask=(self.denom==0).squeeze()
        self.prune_points(prune_mask)
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def adding_postfix(self, added_xyz, added_features_dc, added_features_rest, added_opacities, added_scaling, added_rotation):
        d = {"added_xyz": added_xyz,
        "added_f_dc": added_features_dc,
        "added_f_rest": added_features_rest,
        "added_opacity": added_opacities,
        "added_scaling" : added_scaling,
        "added_rotation" : added_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._added_xyz = optimizable_tensors["added_xyz"]
        self._added_features_dc = optimizable_tensors["added_f_dc"]
        self._added_features_rest = optimizable_tensors["added_f_rest"]
        self._added_opacity = optimizable_tensors["added_opacity"]
        self._added_scaling = optimizable_tensors["added_scaling"]
        self._added_rotation = optimizable_tensors["added_rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.color_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        added_mask=torch.zeros((self.get_xyz.shape[0]), device="cuda", dtype=torch.bool)
        added_mask[-self._added_xyz.shape[0]:]=True
        self._added_mask=added_mask
        
    def adding_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.adding_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def adding_and_split(self, grads, grad_threshold, std_scale, num_of_split=1):
        # Extract points that satisfy the gradient condition
        contracted_xyz=self.get_contracted_xyz()                          
        mask = (contracted_xyz >= 0) & (contracted_xyz <= 1)
        mask = mask.all(dim=1)
        num_of_split=num_of_split
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask, mask)
        stds = std_scale*self.get_scaling[selected_pts_mask].repeat(num_of_split,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self.get_rotation[selected_pts_mask]).repeat(num_of_split,1,1)
        
        added_xyz = (torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(num_of_split, 1)).detach().requires_grad_(True)
        added_scaling = (self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(num_of_split,1) / (0.8*num_of_split))).detach().requires_grad_(True)
        added_rotation = (self.get_rotation[selected_pts_mask].repeat(num_of_split,1)).detach().requires_grad_(True)
        added_features_dc = (self.get_features[:,0:1,:][selected_pts_mask].repeat(num_of_split,1,1)).detach().requires_grad_(True)
        added_features_rest = (self.get_features[:,1:,:][selected_pts_mask].repeat(num_of_split,1,1)).detach().requires_grad_(True)
        added_opacity = (self.inverse_opacity_activation(self.get_opacity[selected_pts_mask]).repeat(num_of_split,1)).detach().requires_grad_(True)

        self.adding_postfix(added_xyz, added_features_dc, added_features_rest, added_opacity, added_scaling, added_rotation)

    def adding_and_prune(self, training_args, extent):
        ## at this point xyz_grad_accum is 0 and denom is 0 
        #print(f"Mean of xyz_grad_accum: {self.xyz_gradient_accum.mean(dim=0)}, first value: {self.xyz_gradient_accum[0]}")
        #print(f"Mean of denom: {self.denom.mean(dim=0)}, first value: {self.denom[0]}")
        grads = self.xyz_gradient_accum / self.denom
        #print(f"How many nans: {grads.isnan().any()}")
        grads[grads.isnan()] = 0.0
        if training_args.s2_adding:
        #    grad_mags = torch.norm(grads, dim=-1)
        #    threshold = torch.quantile(grad_mags, 0.90)  # top 10%
            # to prohibit the threshold to be 0
        #    if threshold == 0:
        #        threshold = training_args.densify_grad_threshold
        #    print(f"Thereshold: {threshold}")

            #threshold = threshold * (1 - iteration / 150)
            #self.adding_and_split(grads, threshold, training_args.std_scale, training_args.num_of_split)
            self.adding_and_split(grads, training_args.densify_grad_threshold, training_args.std_scale, training_args.num_of_split)
        self.prune_added_points(training_args.min_opacity, extent)

        torch.cuda.empty_cache()

    def prune_added_points(self, min_opacity, extent):
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        #print(f"Pruning added points, mask sum: {prune_mask.sum().item()}")
        big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
        #print(f"Big points in world space, mask sum: {big_points_ws.sum().item()}")
        #print(f"What is in - self.added: {-self._added_xyz.shape[0]}")
        ## old
        #prune_mask = torch.logical_or(prune_mask, big_points_ws)[-self._added_xyz.shape[0]:]
        ### Mine
        if self._added_xyz.shape[0] > 0:
            prune_mask = torch.logical_or(prune_mask, big_points_ws)[-self._added_xyz.shape[0]:]
        else:
            prune_mask = torch.zeros(0, dtype=torch.bool, device=prune_mask.device)
        print(f"Pruning added points, mask sum: {prune_mask.sum().item()}")
        if prune_mask.sum() != 0:
            valid_points_mask = ~prune_mask
            #print(f"Valid points mask sum: {valid_points_mask.sum().item()}")
            optimizable_tensors = self._prune_optimizer(valid_points_mask)

            self._added_xyz = optimizable_tensors["added_xyz"]
            self._added_features_dc = optimizable_tensors["added_f_dc"]
            self._added_features_rest = optimizable_tensors["added_f_rest"]
            self._added_opacity = optimizable_tensors["added_opacity"]
            self._added_scaling = optimizable_tensors["added_scaling"]
            self._added_rotation = optimizable_tensors["added_rotation"]
            
            self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
            self.color_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
            self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
            self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

            added_mask=torch.zeros((self.get_xyz.shape[0]), device="cuda", dtype=torch.bool)
            added_mask[-self._added_xyz.shape[0]:]=True
            self._added_mask=added_mask
            torch.cuda.empty_cache()

    def compute_distance_map(self,gt_image: torch.Tensor, rendered_image: torch.Tensor, threshold: float = 0.0) -> torch.Tensor:
        """
        Compute a normalized distance map between the ground truth and rendered image.

        Args:
            gt_image (torch.Tensor): Ground truth image, shape [3, H, W]
            rendered_image (torch.Tensor): Rendered image, shape [3, H, W]
            threshold (float): Small threshold to ignore minor differences.

        Returns:
            torch.Tensor: Normalized distance map of shape [H, W], values ∈ [0, 1]
        """
        assert gt_image.shape == rendered_image.shape and gt_image.shape[0] == 3, "Images must be 3-channel and same shape"
        
        # L2 distance per pixel across color channels
        diff = (gt_image - rendered_image).pow(2).sum(dim=0).sqrt()  # shape: [H, W]
        
        # Apply thresholding and normalization
        diff = torch.clamp(diff - threshold, min=0)
        norm_diff = diff / (diff.max() + 1e-6)  # Avoid division by 0

        return norm_diff  # shape [H, W]


    def spawn_points_color_distance_function(self, gt_image, image, i, viewpoint_cam, visibility_filter, name_of_exp, training_args, args):
        import os
        import torch
        import torchvision.transforms.functional as TF
        import numpy as np
        from scipy.spatial import cKDTree
        from matplotlib.colors import Normalize

        #old_xyz = self.get_xyz.clone()

        grads = self.xyz_gradient_accum[visibility_filter] / self.denom[visibility_filter]
        grads[grads.isnan()] = 0.0
        high_grad_mask = torch.norm(grads, dim=-1) >= training_args.densify_grad_threshold

        # === Step 1: Compute and save distance map ===
        distance_map_norm = self.compute_distance_map(gt_image, image, threshold=0.2)

        # distance_img = (distance_map_norm * 255).byte().cpu()
        # img_pil = TF.to_pil_image(distance_img.unsqueeze(0))
        # os.makedirs("output_dist_maps", exist_ok=True)
        # img_pil.save(f'output_dist_maps/{i}_distance_map.png')

        ### ---- save heatmap of distance map ----
        # distance_map_norm: torch.Tensor of shape [H, W], values in [0, 1]
        distance_map_np = distance_map_norm.detach().cpu().numpy()

        # Create output folder
        os.makedirs(f"{name_of_exp[0]}/output_dist_maps", exist_ok=True)

        # # Save heatmap
        # plt.figure(figsize=(6, 6))
        # plt.axis('off')
        # plt.imshow(distance_map_np, cmap='hot')  # Or 'jet', 'inferno', 'plasma'
        # plt.colorbar()
        # plt.tight_layout()
        # plt.savefig(f'{name_of_exp[0]}/output_dist_maps/{i}_distance_map_heatmap.png', bbox_inches='tight', pad_inches=0)
        # plt.close()

        # Get top-k highest value coordinates
        top_k = 5
        flat_indices = np.argpartition(distance_map_np.flatten(), -top_k)[-top_k:]
        top_y, top_x = np.unravel_index(flat_indices, distance_map_np.shape)

        # Create output folder
        os.makedirs(f"{name_of_exp[0]}/output_dist_maps", exist_ok=True)

        # Save heatmap with annotations
        plt.figure(figsize=(6, 6))
        plt.axis('off')
        plt.imshow(distance_map_np, cmap='hot')
        plt.colorbar()

        # Plot top-k points
        for x, y in zip(top_x, top_y):
            plt.scatter(x, y, s=100, edgecolors='white', facecolors='none', linewidths=1.5)

        plt.tight_layout()
        plt.savefig(f'{name_of_exp[0]}/output_dist_maps/{i}_distance_map_heatmap.png', bbox_inches='tight', pad_inches=0)
        plt.close()

        #print(f"DISTANCE MAP NORM HIGHEST VALUE IS: {distance_map_norm.max().item()}")

        # === Step 2: Project Gaussians to screen ===
        def project_xyz_to_screen(xyz, camera):
            N = xyz.shape[0]
            xyz_h = torch.cat([xyz, torch.ones((N, 1), device=xyz.device)], dim=1)
            screen_coord = (camera.full_proj_transform.T @ xyz_h.T).T
            #print(f"screen cord first: {screen_coord[0]}")
            screen_coord = screen_coord [:, :3] / (screen_coord[:, 3:4])
            #print(f"xyz of first el: {screen_coord[0, :3]}, div: {screen_coord[0, 3]}")
            #print(f"Shape of screen_coord: {screen_coord.shape}")   
            x_ndc, y_ndc = screen_coord[:, 0], screen_coord[:, 1]
            W, H = camera.image_width, camera.image_height
            x = ((x_ndc + 1) / 2 * W).long().clamp(0, W - 1)
            y = ((y_ndc + 1) / 2 * H).long().clamp(0, H - 1)
            return x, y

        # Step 3: Project all Gaussians and check distance map
        # get only visible gaussians from the camera

        xyz = self.get_xyz#[visibility_filter]
        old_xyz = xyz.clone()
        x_pix, y_pix = project_xyz_to_screen(xyz, viewpoint_cam)
        #print(f"Resolution of the normalized distance map: {distance_map_norm.shape}")
        #print(f"How many unqiue pairs are returend in (x_pix, y_pix): {len(set(zip(x_pix.tolist(), y_pix.tolist())))}")
        #distance_values = distance_map_norm[y_pix, x_pix]  # [N]

        # Create green base image (e.g., value 0.3 as "greenish" in grayscale)
        distance_values = torch.full_like(distance_map_norm, fill_value=0.3)
        #print(f"Shape of distance_values: {distance_values.shape}")

        # Make sure x_pix and y_pix are valid and not out of bounds
        mask_valid = (x_pix >= 0) & (x_pix < distance_map_norm.shape[1]) & \
                    (y_pix >= 0) & (y_pix < distance_map_norm.shape[0])

        # Filter valid indices
        x_valid = x_pix[mask_valid]
        y_valid = y_pix[mask_valid]
        dist_vals = distance_map_norm[y_valid, x_valid]

        # Fill only valid positions
        distance_values[y_valid, x_valid] = dist_vals

        sparse_np = distance_values.detach().cpu().numpy()
        masked = np.ma.masked_invalid(sparse_np)

        plt.figure(figsize=(6, 6))
        plt.axis('off')
        plt.imshow(masked, cmap='hot', norm=Normalize(vmin=0.0, vmax=1.0))
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(f"{name_of_exp[0]}/output_dist_maps/{i}_sparse_projected_distance_map.png", bbox_inches='tight', pad_inches=0)
        plt.close()
        #print(f"Is there any distance values equal to 0: {torch.any(distance_values == 0).item()}")
        #print(f"Highest distance value: {distance_values.max().item()}")

        ##### ##### default read from the difference map ##### #####
        
        if not args.is_acc:
            distance_values = distance_map_norm[y_pix, x_pix]  # [N]
            nonzero_mask = distance_values > 0  # Only consider Gaussians in errorful regions

        ##### #####
        def farthest_point_sampling(points, errors, K):
            """
            Select K spatially distant points weighted by error.
            points: [M, 2], errors: [M], both on CPU or GPU
            Returns: [K] indices of selected anchors
            """
            device = points.device
            N = points.shape[0]
            selected = torch.zeros(K, dtype=torch.long, device=device)
            distances = torch.full((N,), float('inf'), device=device)

            # Start with highest-error point
            selected[0] = errors.argmax()
            for i in range(1, K):
                cur_point = points[selected[i - 1]]
                dist = ((points - cur_point) ** 2).sum(dim=1)
                distances = torch.minimum(distances, dist)
                selected[i] = distances.argmax()
            return selected

        # error_coords: [M, 2] from earlier
        error_mask = distance_map_norm > 0  # [H, W]
        error_y, error_x = torch.nonzero(error_mask, as_tuple=True)  # [M], [M]
        error_coords = torch.stack([error_x, error_y], dim=1).float()  # [M, 2]
        error_values = distance_map_norm[error_y, error_x]  # [M]
        gaussian_coords = torch.stack([x_pix, y_pix], dim=1).float()  # [N, 2]

        # Sample K anchors (e.g., 100)
        K = 100
        anchor_indices = farthest_point_sampling(error_coords, error_values, K)
        anchor_coords = error_coords[anchor_indices]  # [K, 2]

        # Assign each Gaussian to its nearest anchor
        # gaussian_coords: [N, 2]
        anchor_dists = torch.cdist(gaussian_coords, anchor_coords)  # [N, K]
        assigned_anchor = anchor_dists.argmin(dim=1)  # [N]

        assigned_errors = torch.zeros(len(gaussian_coords), device=gaussian_coords.device)

        for i, anchor in enumerate(anchor_coords):
            # Find Gaussians near this anchor
            nearby_mask = assigned_anchor == i
            if not nearby_mask.any():
                continue

            nearby_gaussians = gaussian_coords[nearby_mask]  # [N', 2]
            anchor_repeated = anchor.unsqueeze(0).expand(nearby_gaussians.size(0), -1)  # [N', 2]

            dists = ((nearby_gaussians - anchor_repeated) ** 2).sum(dim=1)
            # You can threshold or weight here:
            assigned_errors[nearby_mask] = torch.exp(-dists)  # Or use 1 / (1 + dists), etc.

        # eliminate those gaussians which have erro = 0
        if args.is_acc:
            nonzero_mask = assigned_errors > 0  # Keep only Gaussians with assigned errors
        # You now have: assigned_errors[i] = number of errorful pixels assigned to Gaussian i
        # === Initialize Heatmap ===
        H, W = distance_map_norm.shape
        heatmap = torch.zeros((H, W), device=assigned_errors.device)

        # === Accumulate Errors at Gaussian Screen Positions ===
        heatmap.index_put_((y_pix, x_pix), assigned_errors, accumulate=True)

        # === Normalize to [0,1] ===
        heatmap_np = heatmap.detach().cpu().numpy()
        heatmap_np /= (heatmap_np.max() + 1e-6)

        # === Optional: Mask non-Gaussian areas (e.g., black or green background) ===
        # This helps visualize only the areas covered by Gaussians.
        visible_mask = (heatmap_np > 0)
        colored_heatmap = plt.cm.hot(heatmap_np)  # shape (H, W, 4)

        # Set background to black where there are no Gaussians
        colored_heatmap[~visible_mask] = [1, 1, 1, 1]  # black with full alpha

        # === Save the result ===
        plt.figure(figsize=(8, 8))
        plt.axis('off')
        plt.imshow(colored_heatmap)
        plt.tight_layout()
        plt.savefig(f'{name_of_exp[0]}/output_dist_maps/{i}_gaussian_only_error_heatmap.png', bbox_inches='tight', pad_inches=0)
        plt.close()

        ##### visualize distance map norm as heatmap #####
        H, W = distance_map_norm.shape

        # 2. Initialize heatmap to NaN for better visualization
        heatmap = torch.full((H, W), float('nan'), device=distance_map_norm.device)

        # 3. Scatter valid distance values
        heatmap[y_pix, x_pix] = distance_map_norm[y_pix, x_pix]
        # ignore the values equal to 0
        heatmap[heatmap == 0] = float('nan')  # Ignore zero values in the heatmap

        # 4. Convert to NumPy
        heatmap_np = heatmap.detach().cpu().numpy()

        plt.figure(figsize=(6, 6))
        plt.axis('off')
        im = plt.imshow(heatmap_np, cmap='hot', interpolation='nearest')
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(f"{name_of_exp[0]}/output_dist_maps/{i}_distance_heatmap.png", bbox_inches='tight', pad_inches=0)
        plt.close()
        #####

        #print(f"Number of Gaussians in errorful regions: {nonzero_mask.sum().item()}")
        #print(f"Min max distance values: {distance_values.min().item()}, {distance_values.max().item()}")
        if nonzero_mask.sum() == 0:
            print("No non-zero pixels in distance map for any Gaussians.")
            return torch.zeros_like(nonzero_mask)


        selected_pts_mask = nonzero_mask #& high_grad_mask

        # selected_xyz = self.get_xyz[selected_pts_mask]
        # if selected_xyz.shape[0] > 0:
        #     all_xyz_np = self.get_xyz.detach().cpu().numpy()
        #     tree = cKDTree(all_xyz_np)
        #     selected_xyz_np = selected_xyz.detach().cpu().numpy()
        #     radius = 0.02
        #     local_density = torch.tensor([len(tree.query_ball_point(x, radius)) for x in selected_xyz_np], device=self.get_xyz.device)
        #     density_mask = local_density <= 50
        #     full_density_mask = torch.zeros_like(selected_pts_mask)
        #     selected_indices = selected_pts_mask.nonzero(as_tuple=False).squeeze(1)
        #     full_density_mask[selected_indices[density_mask]] = True
        #     selected_pts_mask &= full_density_mask

        # === Step 5: Spawn
        final_indices = selected_pts_mask.nonzero(as_tuple=False).squeeze(1)
        spawn_counts = (1 + distance_map_norm[y_pix[final_indices], x_pix[final_indices]] * args.max_spawn_count).round().long()

        selected_xyz = self.get_xyz[final_indices]
        selected_scaling = self.get_scaling[final_indices]
        selected_rotation = self.get_rotation[final_indices]
        selected_feat_dc = self.get_features[:, 0:1, :][final_indices]
        selected_feat_rest = self.get_features[:, 1:, :][final_indices]

        if spawn_counts.sum().item() == 0:
            print("No Gaussians to spawn after filtering. Spawning one randomly.")

            # === Random spawn fallback ===
            random_index = torch.randint(0, self.get_xyz.shape[0], (1,), device=self.get_xyz.device)
            selected_xyz = self.get_xyz[random_index]
            selected_scaling = self.get_scaling[random_index]
            selected_rotation = self.get_rotation[random_index]
            selected_feat_dc = self.get_features[:, 0:1, :][random_index]
            selected_feat_rest = self.get_features[:, 1:, :][random_index]

            # Spawn one Gaussian with random perturbation
            stds = training_args.std_scale * selected_scaling
            samples = torch.normal(mean=torch.zeros_like(stds), std=stds)
            rots = build_rotation(selected_rotation)
            new_xyz = (torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + selected_xyz).detach().requires_grad_(True)

            self._added_xyz = new_xyz
            self._added_scaling = self.scaling_inverse_activation(selected_scaling / (0.8 * args.max_spawn_count)).detach().requires_grad_(True)
            self._added_rotation = torch.tensor([[1., 0., 0., 0.]], device='cuda').detach().requires_grad_(True)
            self._added_features_dc = selected_feat_dc.detach().requires_grad_(True)
            self._added_features_rest = selected_feat_rest.detach().requires_grad_(True)
            self._added_opacity = self.inverse_opacity_activation(torch.tensor([0.1], device='cuda')).unsqueeze(0).detach().requires_grad_(True)

            return torch.zeros_like(spawn_counts, dtype=torch.bool, device=self.get_xyz.device)

        xyz_base = selected_xyz.repeat_interleave(spawn_counts, dim=0)
        scaling_base = selected_scaling.repeat_interleave(spawn_counts, dim=0)
        rotation_base = selected_rotation.repeat_interleave(spawn_counts, dim=0)
        feat_dc_base = selected_feat_dc.repeat_interleave(spawn_counts, dim=0)
        feat_rest_base = selected_feat_rest.repeat_interleave(spawn_counts, dim=0)

        stds = training_args.std_scale * scaling_base
        samples = torch.normal(mean=torch.zeros_like(stds), std=stds)
        rots = build_rotation(rotation_base)
        new_xyz = (torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + xyz_base).detach().requires_grad_(True)
        #print(f"[SPAWN] New XYZ shape: {new_xyz.shape}")

        total_spawns = spawn_counts.sum().item()
        self._added_xyz = new_xyz
        self._added_scaling = self.scaling_inverse_activation(scaling_base / (0.8 * args.max_spawn_count)).detach().requires_grad_(True)
        self._added_rotation = torch.tensor([1., 0., 0., 0.], device='cuda').repeat(total_spawns, 1).detach().requires_grad_(True)
        self._added_features_dc = feat_dc_base.detach().requires_grad_(True)
        self._added_features_rest = feat_rest_base.detach().requires_grad_(True)
        self._added_opacity = self.inverse_opacity_activation(torch.tensor([0.1], device='cuda')).repeat(total_spawns, 1).detach().requires_grad_(True)
        #print(f"Shape of added_opacities: {self._added_opacity.shape}")

        percentage = 100 * final_indices.shape[0] / len(self.get_xyz)
        print(f"[SPAWN] Selected Gaussians: {final_indices.shape[0]}/{len(self.get_xyz)} ({percentage:.2f}%), New Gaussians Spawned: {total_spawns}")
        #print(f"Top distance found: {distance_map_norm[y_pix[final_indices], x_pix[final_indices]].max().item()}")
        # indentify idx of highest spawn count gaussian
        highest_spawn_count_idx = spawn_counts.argmax().item()
        #print(f"Distance for the highest spawn count gaussians after both filters: {distance_map_norm[y_pix[final_indices[highest_spawn_count_idx]], x_pix[final_indices[highest_spawn_count_idx]]].item()}")
        return selected_pts_mask, spawn_counts, old_xyz

    def spawn_points_color_distance_multiview(
        self, gt_images, images, cams, masked_i, visibility_filter, name_of_exp, training_args, args
    ):
        """
        Args:
            gt_images: list of [H, W, 3] ground truth images (Torch or NumPy)
            images: list of [H, W, 3] predicted images
            cams: list of COLMAP-style camera objects with .full_proj_transform, .image_width, .image_height
        """
        import torch
        import numpy as np

        num_views = len(gt_images)
        assert len(images) == len(cams) == num_views

        old_xyz = self.get_xyz.clone() 

        # Initialize scores
        aggregated_error = torch.zeros(self.get_xyz.shape[0], device=self.get_xyz.device)
        aggregated_counts = torch.zeros_like(aggregated_error)

        print(f"Number of views: {num_views}")

        for i in range(num_views):
            # Step 1: compute per-view distance map
            distance_map = self.compute_distance_map(gt_images[i], images[i], threshold=0.2)  # [H, W]

            # Step 2: project Gaussians to this view
            x_pix, y_pix = self.project_xyz_to_screen(self.get_xyz, cams[i])  # [N]

            H, W = distance_map.shape
            mask_valid = (x_pix >= 0) & (x_pix < W) & (y_pix >= 0) & (y_pix < H)
            x_valid = x_pix[mask_valid]
            y_valid = y_pix[mask_valid]
            gaussian_indices = torch.nonzero(mask_valid, as_tuple=False).squeeze(1)

            # Step 3: assign errors from this view
            view_errors = distance_map[y_valid, x_valid]  # [M]
            aggregated_error[gaussian_indices] += view_errors
            aggregated_counts[gaussian_indices] += 1

        # Step 4: average error across views
        valid = aggregated_counts > 0
        aggregated_error[valid] /= aggregated_counts[valid]

        # Step 5: spawn mask based on error
        spawn_mask = aggregated_error.clone()
        spawn_mask[spawn_mask < 0.1] = 0.0  # Or threshold adaptively

        # Step 1: Project Gaussians to the selected view
        x_pix, y_pix = self.project_xyz_to_screen(self.get_xyz, cams[0])  # [N]

        # Step 2: Clamp coordinates and build valid mask
        H, W = gt_images[0].shape[-2:]  # use shape of a ground truth image
        mask_valid = (x_pix >= 0) & (x_pix < W) & (y_pix >= 0) & (y_pix < H)
        mask_nonzero = spawn_mask > 0  # [N]
        mask_total = mask_valid & mask_nonzero  # [N]
        x_valid = x_pix[mask_total]
        y_valid = y_pix[mask_total]
        valid_errors = spawn_mask[mask_total]  # [M]

        # Step 3: Normalize error values (optional but recommended)
        valid_errors_norm = (valid_errors - valid_errors.min()) / (valid_errors.max() - valid_errors.min() + 1e-6)


        # Step 4: Create a blank heatmap and fill with values at projected positions
        heatmap = torch.zeros((H, W), dtype=torch.float32, device=spawn_mask.device)
        heatmap[y_valid, x_valid] = valid_errors_norm

        # Step 5: Mask out zero values for cleaner visualization
        heatmap[heatmap == 0] = float('nan')

        # Step 6: Convert to NumPy and plot
        heatmap_np = heatmap.detach().cpu().numpy()
        os.makedirs(f"{name_of_exp[0]}/output_dist_maps", exist_ok=True)

        plt.figure(figsize=(6, 6))
        plt.axis('off')
        im = plt.imshow(heatmap_np, cmap='hot', interpolation='nearest')
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(f"{name_of_exp[0]}/output_dist_maps/{masked_i}_aggregated_error_heatmap.png", bbox_inches='tight', pad_inches=0)
        plt.close()


        final_indices = spawn_mask.nonzero(as_tuple=False).squeeze(1)
        selected_pts_mask = torch.zeros_like(spawn_mask, dtype=torch.bool)
        selected_pts_mask[final_indices] = True

        # Use the same spawn logic you already have:
        spawn_counts = (1 + aggregated_error[final_indices] * args.max_spawn_count).round().long()

        selected_xyz = self.get_xyz[final_indices]
        selected_scaling = self.get_scaling[final_indices]
        selected_rotation = self.get_rotation[final_indices]
        selected_feat_dc = self.get_features[:, 0:1, :][final_indices]
        selected_feat_rest = self.get_features[:, 1:, :][final_indices]

        if spawn_counts.sum().item() == 0:
            print("No Gaussians to spawn after filtering. Spawning one randomly.")

            # === Random spawn fallback ===
            random_index = torch.randint(0, self.get_xyz.shape[0], (1,), device=self.get_xyz.device)
            selected_xyz = self.get_xyz[random_index]
            selected_scaling = self.get_scaling[random_index]
            selected_rotation = self.get_rotation[random_index]
            selected_feat_dc = self.get_features[:, 0:1, :][random_index]
            selected_feat_rest = self.get_features[:, 1:, :][random_index]

            # Spawn one Gaussian with random perturbation
            stds = training_args.std_scale * selected_scaling
            samples = torch.normal(mean=torch.zeros_like(stds), std=stds)
            rots = build_rotation(selected_rotation)
            new_xyz = (torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + selected_xyz).detach().requires_grad_(True)

            self._added_xyz = new_xyz
            self._added_scaling = self.scaling_inverse_activation(selected_scaling / (0.8 * args.max_spawn_count)).detach().requires_grad_(True)
            self._added_rotation = torch.tensor([[1., 0., 0., 0.]], device='cuda').detach().requires_grad_(True)
            self._added_features_dc = selected_feat_dc.detach().requires_grad_(True)
            self._added_features_rest = selected_feat_rest.detach().requires_grad_(True)
            self._added_opacity = self.inverse_opacity_activation(torch.tensor([0.1], device='cuda')).unsqueeze(0).detach().requires_grad_(True)

            return torch.zeros_like(spawn_counts, dtype=torch.bool, device=self.get_xyz.device)

        xyz_base = selected_xyz.repeat_interleave(spawn_counts, dim=0)
        scaling_base = selected_scaling.repeat_interleave(spawn_counts, dim=0)
        rotation_base = selected_rotation.repeat_interleave(spawn_counts, dim=0)
        feat_dc_base = selected_feat_dc.repeat_interleave(spawn_counts, dim=0)
        feat_rest_base = selected_feat_rest.repeat_interleave(spawn_counts, dim=0)

        stds = training_args.std_scale * scaling_base
        samples = torch.normal(mean=torch.zeros_like(stds), std=stds)
        rots = build_rotation(rotation_base)
        new_xyz = (torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + xyz_base).detach().requires_grad_(True)
        #print(f"[SPAWN] New XYZ shape: {new_xyz.shape}")

        total_spawns = spawn_counts.sum().item()
        self._added_xyz = new_xyz
        self._added_scaling = self.scaling_inverse_activation(scaling_base / (0.8 * args.max_spawn_count)).detach().requires_grad_(True)
        self._added_rotation = torch.tensor([1., 0., 0., 0.], device='cuda').repeat(total_spawns, 1).detach().requires_grad_(True)
        self._added_features_dc = feat_dc_base.detach().requires_grad_(True)
        self._added_features_rest = feat_rest_base.detach().requires_grad_(True)
        self._added_opacity = self.inverse_opacity_activation(torch.tensor([0.1], device='cuda')).repeat(total_spawns, 1).detach().requires_grad_(True)
        #print(f"Shape of added_opacities: {self._added_opacity.shape}")

        percentage = 100 * final_indices.shape[0] / len(self.get_xyz)
        print(f"[SPAWN] Selected Gaussians: {final_indices.shape[0]}/{len(self.get_xyz)} ({percentage:.2f}%), New Gaussians Spawned: {total_spawns}")
        #print(f"Top distance found: {distance_map_norm[y_pix[final_indices], x_pix[final_indices]].max().item()}")
        # indentify idx of highest spawn count gaussian
        #highest_spawn_count_idx = spawn_counts.argmax().item()
        #print(f"Distance for the highest spawn count gaussians after both filters: {distance_map_norm[y_pix[final_indices[highest_spawn_count_idx]], x_pix[final_indices[highest_spawn_count_idx]]].item()}")

        self.save_distance_map_visualizations(
            distance_map, x_pix, y_pix, masked_i, name_of_exp
        )
        return selected_pts_mask, spawn_counts, old_xyz
    


    def save_distance_map_visualizations(self, distance_map_norm, x_pix, y_pix, i, name_of_exp):
        import os
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize

        name_of_exp = name_of_exp[0]

        os.makedirs(f"{name_of_exp}/output_dist_maps", exist_ok=True)

        # Save raw heatmap
        distance_map_np = distance_map_norm.detach().cpu().numpy()
        plt.figure(figsize=(6, 6))
        plt.axis('off')
        plt.imshow(distance_map_np, cmap='hot')
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(f"{name_of_exp}/output_dist_maps/{i}_distance_map_heatmap.png", bbox_inches='tight', pad_inches=0)
        plt.close()

        # Save annotated heatmap with top-k values
        top_k = 5
        flat_indices = np.argpartition(distance_map_np.flatten(), -top_k)[-top_k:]
        top_y, top_x = np.unravel_index(flat_indices, distance_map_np.shape)

        plt.figure(figsize=(6, 6))
        plt.axis('off')
        plt.imshow(distance_map_np, cmap='hot')
        plt.colorbar()
        for x, y in zip(top_x, top_y):
            plt.scatter(x, y, s=100, edgecolors='white', facecolors='none', linewidths=1.5)
        plt.tight_layout()
        plt.savefig(f"{name_of_exp}/output_dist_maps/{i}_distance_map_topk.png", bbox_inches='tight', pad_inches=0)
        plt.close()

        # Projected values only
        distance_values = torch.full_like(distance_map_norm, float('nan')) # filled as white 
        mask_valid = (x_pix >= 0) & (x_pix < distance_map_norm.shape[1]) & (y_pix >= 0) & (y_pix < distance_map_norm.shape[0])
        x_valid = x_pix[mask_valid]
        y_valid = y_pix[mask_valid]
        dist_vals = distance_map_norm[y_valid, x_valid]
        distance_values[y_valid, x_valid] = dist_vals
        sparse_np = distance_values.detach().cpu().numpy()
        masked = np.ma.masked_invalid(sparse_np)

        plt.figure(figsize=(6, 6))
        plt.axis('off')
        plt.imshow(masked, cmap='hot', vmin=np.nanmin(masked), vmax=np.nanmax(masked))
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(f"{name_of_exp}/output_dist_maps/{i}_sparse_projected_distance_map.png", bbox_inches='tight', pad_inches=0)
        plt.close()

        # Final projected heatmap ignoring zero values
        heatmap = torch.full_like(distance_map_norm, float('nan'))
        heatmap[y_pix, x_pix] = distance_map_norm[y_pix, x_pix]
        heatmap[heatmap == 0] = float('nan')
        heatmap_np = heatmap.detach().cpu().numpy()

        plt.figure(figsize=(6, 6))
        plt.axis('off')
        im = plt.imshow(heatmap_np, cmap='hot', interpolation='nearest')
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(f"{name_of_exp}/output_dist_maps/{i}_distance_heatmap.png", bbox_inches='tight', pad_inches=0)
        plt.close()

    # === Step 2: Project Gaussians to Screen ===
    def project_xyz_to_screen(self, xyz, camera):
        N = xyz.shape[0]
        xyz_h = torch.cat([xyz, torch.ones((N, 1), device=xyz.device)], dim=1)
        screen_coord = (camera.full_proj_transform.T @ xyz_h.T).T
        screen_coord = screen_coord[:, :3] / (screen_coord[:, 3:4])
        x_ndc, y_ndc = screen_coord[:, 0], screen_coord[:, 1]
        W, H = camera.image_width, camera.image_height
        x_pix = ((x_ndc + 1) / 2 * W).long().clamp(0, W - 1)
        y_pix = ((y_ndc + 1) / 2 * H).long().clamp(0, H - 1)
        return x_pix, y_pix

    def spawn_constant_from_distance(self, gt_image, rendered_image, viewpoint_cam, i, name_of_exp, training_args, args):
        import os
        import torch
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize

        # === Step 1: Compute Distance Map ===
        distance_map_norm = self.compute_distance_map(gt_image, rendered_image, threshold=0.6)

        xyz = self.get_xyz
        old_xyz = xyz.clone()
        x_pix, y_pix = self.project_xyz_to_screen(xyz, viewpoint_cam)

        # === Step 3: Select Gaussians With Error ===
        distance_values = distance_map_norm[y_pix, x_pix]
        selected_pts_mask = distance_values > 0
        final_indices = selected_pts_mask.nonzero(as_tuple=False).squeeze(1)

        if final_indices.shape[0] == 0:
            print("[SPAWN] No Gaussians found with color error.")
            grads = self.xyz_gradient_accum / self.denom
            grads[grads.isnan()] = 0.0

            #grad_norms = torch.norm(grads, dim=-1)
            #print(f"Grad min: {grad_norms.min().item()}, mean: {grad_norms.mean().item()}, max: {grad_norms.max().item()}")
            #threshold = grad_norms.quantile(0.8).item()

            contracted_xyz=self.get_contracted_xyz()                          
            mask = (contracted_xyz >= 0) & (contracted_xyz <= 1)
            mask = mask.all(dim=1)
            
            print("Spawning points gradient-based as fallback.")
            return self.spawn_points_gradient_based(grads, training_args, mask=mask, max_spawn_count=args.num_of_spawn)

        # === Step 4: Constant Spawn Count ===
        num_spawn = args.num_of_spawn
        spawn_counts = torch.full((final_indices.shape[0],), num_spawn, dtype=torch.long, device=xyz.device)

        # === Step 5: Gather Parameters for Selected Points ===
        selected_xyz = self.get_xyz[final_indices]
        selected_scaling = self.get_scaling[final_indices]
        selected_rotation = self.get_rotation[final_indices]
        selected_feat_dc = self.get_features[:, 0:1, :][final_indices]
        selected_feat_rest = self.get_features[:, 1:, :][final_indices]

        # === Step 6: Prepare for Spawning ===
        xyz_base = selected_xyz.repeat_interleave(spawn_counts, dim=0)
        scaling_base = selected_scaling.repeat_interleave(spawn_counts, dim=0)
        rotation_base = selected_rotation.repeat_interleave(spawn_counts, dim=0)
        feat_dc_base = selected_feat_dc.repeat_interleave(spawn_counts, dim=0)
        feat_rest_base = selected_feat_rest.repeat_interleave(spawn_counts, dim=0)


        stds = training_args.std_scale * scaling_base
        samples = torch.normal(mean=torch.zeros_like(stds), std=stds)
        rots = build_rotation(rotation_base)
        new_xyz = (torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + xyz_base).detach().requires_grad_(True)

        # === Step 7: Store as "Added" Gaussians ===
        self._added_xyz = new_xyz
        self._added_scaling = self.scaling_inverse_activation(scaling_base / (0.8 * num_spawn)).detach().requires_grad_(True)
        self._added_rotation = torch.tensor([1., 0., 0., 0.], device='cuda').repeat(new_xyz.shape[0], 1).detach().requires_grad_(True)
        self._added_features_dc = feat_dc_base.detach().requires_grad_(True)
        self._added_features_rest = feat_rest_base.detach().requires_grad_(True)
        self._added_opacity = self.inverse_opacity_activation(torch.tensor([0.1], device='cuda')).repeat(new_xyz.shape[0], 1).detach().requires_grad_(True)

        print(f"[SPAWN] Spawned {new_xyz.shape[0]} Gaussians from {final_indices.shape[0]} errorful points.")

        self.save_distance_map_visualizations(distance_map_norm, x_pix, y_pix, i, name_of_exp)
        
        return selected_pts_mask, spawn_counts, old_xyz


    def spawn_gaussians_by_error(self, gt_image, rendered_image, viewpoint_cam, training_args, args, i, name_of_exp, visibility_filter):


        # Compute distance/error map between GT and predicted image
        distance_map_norm = self.compute_distance_map(gt_image, rendered_image, threshold=0.4)

        # Project Gaussians to screen
        old_xyz = self.get_xyz.clone()
        x_pix, y_pix = self.project_xyz_to_screen(old_xyz[visibility_filter], viewpoint_cam)

        name_of_exp = name_of_exp[0]
        os.makedirs(f"{name_of_exp}/output_dist_maps", exist_ok=True)

        # Build 2D Gaussian positions
        gaussian_coords = torch.stack([x_pix, y_pix], dim=1).float()

        # Extract errorful pixel positions and values
        error_mask = distance_map_norm > 0.0
        if not error_mask.any():
            print("[SPAWN] No errorful pixels in map.")
            return

        error_y, error_x = torch.nonzero(error_mask, as_tuple=True)
        error_coords = torch.stack([error_x, error_y], dim=1).float()
        error_values = distance_map_norm[error_y, error_x]

        # Assign each error pixel to nearest Gaussian
        dists = torch.cdist(error_coords, gaussian_coords)  # [M, N]
        closest_gaussian = dists.argmin(dim=1)
        accumulated_error = torch.zeros(len(old_xyz), device=old_xyz.device)
        accumulated_error.scatter_add_(0, closest_gaussian, error_values)

        if accumulated_error.max().item() == 0:
            print("[SPAWN] All errors accumulated to 0.")
            return

        # Normalize accumulated error
        normalized_error = accumulated_error / (accumulated_error.max() + 1e-6)

        # === Visualization ===
        plt.figure(figsize=(8, 8))
        plt.imshow(distance_map_norm.detach().cpu().numpy(), cmap='gray', alpha=0.3)
        plt.scatter(x_pix.detach().cpu(), y_pix.detach().cpu(), c=normalized_error.detach().cpu().numpy(), cmap='hot', s=10)
        plt.colorbar(label='Accumulated Error per Gaussian')
        plt.title(f'Error-weighted Gaussian Spawn Map [{i}]')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f"{name_of_exp}/output_dist_maps/{i}_gaussian_error_heatmap.png", bbox_inches='tight', pad_inches=0)
        plt.close()

        # === Spawn Logic ===
        spawn_counts = (normalized_error * args.num_of_spawns).round().long()
        mask = spawn_counts > 0
        selected_indices = mask.nonzero(as_tuple=False).squeeze(1)

        if selected_indices.numel() == 0:
            print("[SPAWN] No Gaussians selected for spawning.")
            return

        selected_xyz = old_xyz[selected_indices]
        selected_scaling = self.get_scaling[selected_indices]
        selected_rotation = self.get_rotation[selected_indices]
        selected_feat_dc = self.get_features[:, 0:1, :][selected_indices]
        selected_feat_rest = self.get_features[:, 1:, :][selected_indices]

        # Expand based on spawn counts
        expanded_xyz = selected_xyz.repeat_interleave(spawn_counts[selected_indices], dim=0)
        expanded_scaling = selected_scaling.repeat_interleave(spawn_counts[selected_indices], dim=0)
        expanded_rotation = selected_rotation.repeat_interleave(spawn_counts[selected_indices], dim=0)
        expanded_dc = selected_feat_dc.repeat_interleave(spawn_counts[selected_indices], dim=0)
        expanded_rest = selected_feat_rest.repeat_interleave(spawn_counts[selected_indices], dim=0)

        # Sample new positions
        stds = training_args.std_scale * expanded_scaling
        samples = torch.normal(mean=torch.zeros_like(stds), std=stds)
        rots = build_rotation(expanded_rotation)
        new_xyz = (torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + expanded_xyz).detach().requires_grad_(True)

        # Register new points
        self._added_xyz = new_xyz
        self._added_scaling = self.scaling_inverse_activation(expanded_scaling / (0.8 * args.num_of_spawns)).detach().requires_grad_(True)
        self._added_rotation = torch.tensor([1., 0., 0., 0.], device='cuda').repeat(new_xyz.shape[0], 1).detach().requires_grad_(True)
        self._added_features_dc = expanded_dc.detach().requires_grad_(True)
        self._added_features_rest = expanded_rest.detach().requires_grad_(True)

        self._added_opacity = self.inverse_opacity_activation(torch.tensor([0.1], device='cuda')).repeat(new_xyz.shape[0], 1).detach().requires_grad_(True)

        print(f"[SPAWN] Spawned {new_xyz.shape[0]} Gaussians based on accumulated error.")
        return selected_indices, spawn_counts[selected_indices], old_xyz



    def spawn_points_gradient_based(self, grads, training_args, mask=None, max_spawn_count=15):
        """
        Spawn Gaussians based on the magnitude of view-space gradients.

        Args:
            grads (Tensor): [N, 3] gradient vectors.
            training_args: configuration object with .densify_grad_threshold and .std_scale.
            mask (Tensor, optional): [N] boolean mask to filter valid Gaussians.
            max_spawn_count (int): maximum number of Gaussians to spawn per source.

        Returns:
            None – modifies self._added_xyz, self._added_scaling, etc.
        """
        grads[grads.isnan()] = 0.0
        grad_mag = torch.norm(grads, dim=-1)

        old_xyz = self.get_xyz.clone()

        selected_pts_mask = grad_mag >= training_args.densify_grad_threshold
        if mask is not None:
            selected_pts_mask &= mask

        selected_xyz = self.get_xyz[selected_pts_mask]
        if selected_xyz.shape[0] == 0:
            print("[SPAWN] No valid points for spawning.")
            return

        grad_vals = grad_mag[selected_pts_mask]
        spawn_counts = (1 + grad_vals / (grad_vals.max() + 1e-6) * max_spawn_count).round().long()

        # Repeat relevant attributes
        xyz_base = selected_xyz.repeat_interleave(spawn_counts, dim=0)
        scaling_base = self.get_scaling[selected_pts_mask].repeat_interleave(spawn_counts, dim=0)
        rotation_base = self.get_rotation[selected_pts_mask].repeat_interleave(spawn_counts, dim=0)
        feat_dc_base = self.get_features[:, 0:1, :][selected_pts_mask].repeat_interleave(spawn_counts, dim=0)
        feat_rest_base = self.get_features[:, 1:, :][selected_pts_mask].repeat_interleave(spawn_counts, dim=0)

        # Generate new positions
        stds = training_args.std_scale * scaling_base
        samples = torch.normal(mean=torch.zeros_like(stds), std=stds)
        rots = build_rotation(rotation_base)
        new_xyz = (torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + xyz_base).detach().requires_grad_(True)

        # Set added fields
        self._added_xyz = new_xyz
        self._added_scaling = self.scaling_inverse_activation(scaling_base / (0.8 * max_spawn_count)).detach().requires_grad_(True)
        self._added_rotation = torch.tensor([1., 0., 0., 0.], device='cuda').repeat(new_xyz.shape[0], 1).detach().requires_grad_(True)
        self._added_features_dc = feat_dc_base.detach().requires_grad_(True)
        self._added_features_rest = feat_rest_base.detach().requires_grad_(True)
        self._added_opacity = self.inverse_opacity_activation(torch.tensor([0.1], device='cuda')).repeat(new_xyz.shape[0], 1).detach().requires_grad_(True)

        print(f"[SPAWN] Spawned {new_xyz.shape[0]} Gaussians from {selected_xyz.shape[0]} sources.")

        return selected_pts_mask, spawn_counts, old_xyz


    def extract_errorful_gaussians_from_diff_map(self, gt_image, rendered_image, viewpoint_cam, args):
        """
        Extracts Gaussian points that contribute to high error in the color difference map,
        without performing any spawning or accumulation.

        Returns:
            selected_pts_mask (BoolTensor): [N] mask of Gaussians with non-zero difference.
            spawn_counts (LongTensor): [M] values ∝ error magnitude from diff map.
            old_xyz (Tensor): [N, 3] cloned Gaussian positions before modification.
        """
        # Step 1: Compute L2 difference map
        distance_map_norm = self.compute_distance_map(gt_image, rendered_image, threshold=0.1)

        # Step 2: Project Gaussian 3D positions to image plane
        def project_xyz_to_screen(xyz, cam):
            N = xyz.shape[0]
            xyz_h = torch.cat([xyz, torch.ones((N, 1), device=xyz.device)], dim=1)
            screen = (cam.full_proj_transform.T @ xyz_h.T).T
            screen = screen[:, :3] / screen[:, 3:4]
            x_ndc, y_ndc = screen[:, 0], screen[:, 1]
            W, H = cam.image_width, cam.image_height
            x_pix = ((x_ndc + 1) / 2 * W).long().clamp(0, W - 1)
            y_pix = ((y_ndc + 1) / 2 * H).long().clamp(0, H - 1)
            return x_pix, y_pix

        xyz = self.get_xyz
        old_xyz = xyz.clone()
        x_pix, y_pix = project_xyz_to_screen(xyz, viewpoint_cam)

        # Step 3: Get color error at projected locations
        distance_values = distance_map_norm[y_pix, x_pix]  # [N]
        nonzero_mask = distance_values > 0
        final_indices = nonzero_mask.nonzero(as_tuple=False).squeeze(1)

        # Use normalized error value to define "spawn count-like" signal
        spawn_counts = (1 + distance_values[final_indices] * args.max_spawn_count).round().long()
        selected_pts_mask = nonzero_mask

        return selected_pts_mask, spawn_counts, old_xyz


    def training_one_frame_s2_setup(self, gt_image, image, i, viewpoint_cam, visibility_filter, name_of_exp, pipe, training_args, args):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        #grad_norms = torch.norm(grads, dim=-1)
        #print(f"Grad min: {grad_norms.min().item()}, mean: {grad_norms.mean().item()}, max: {grad_norms.max().item()}")
        #threshold = grad_norms.quantile(0.8).item()

        contracted_xyz=self.get_contracted_xyz()                          
        mask = (contracted_xyz >= 0) & (contracted_xyz <= 1)
        mask = mask.all(dim=1)
        
        if training_args.spawn_type=='clone':
        # Clone
            selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= training_args.densify_grad_threshold, True, False)
            selected_pts_mask = torch.logical_and(selected_pts_mask, mask)
            self._added_xyz = self.get_xyz[selected_pts_mask].detach().clone().requires_grad_(True)
            self._added_features_dc = self.get_features[:,0:1,:][selected_pts_mask].detach().clone().requires_grad_(True)
            self._added_features_rest = self.get_features[:,1:,:][selected_pts_mask].detach().clone().requires_grad_(True)
            self._added_opacity = self._opacity[selected_pts_mask].detach().clone().requires_grad_(True)
            self._added_scaling = self._scaling[selected_pts_mask].detach().clone().requires_grad_(True)
            self._added_rotation = self.get_rotation[selected_pts_mask].detach().clone().requires_grad_(True)
        
        elif training_args.spawn_type=='split':
        # Split
            num_of_split=training_args.num_of_split
            selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= training_args.densify_grad_threshold, True, False)
            selected_pts_mask = torch.logical_and(selected_pts_mask, mask)
            stds = training_args.std_scale*self.get_scaling[selected_pts_mask].repeat(num_of_split,1)
            means =torch.zeros((stds.size(0), 3),device="cuda")
            samples = torch.normal(mean=means, std=stds)
            rots = build_rotation(self.get_rotation[selected_pts_mask]).repeat(num_of_split,1,1)
            self._added_xyz = (torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(num_of_split, 1)).detach().requires_grad_(True)
            self._added_scaling = (self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(num_of_split,1) / (0.8*num_of_split))).detach().requires_grad_(True)
            self._added_rotation = (self.get_rotation[selected_pts_mask].repeat(num_of_split,1)).detach().requires_grad_(True)
            self._added_features_dc = (self.get_features[:,0:1,:][selected_pts_mask].repeat(num_of_split,1,1)).detach().requires_grad_(True)
            self._added_features_rest = (self.get_features[:,1:,:][selected_pts_mask].repeat(num_of_split,1,1)).detach().requires_grad_(True)
            self._added_opacity = (self._opacity[selected_pts_mask].repeat(num_of_split,1)).detach().requires_grad_(True)

        ## COMEBACK HERE
        elif training_args.spawn_type=='spawn':
        # Spawn
            # # Spawn OLD
            if not args.new_spawn:
                num_of_spawn=training_args.num_of_spawn
                old_xyz = self.get_xyz.clone()
                selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= training_args.densify_grad_threshold, True, False)
                #selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= threshold, True, False)
                selected_pts_mask = torch.logical_and(selected_pts_mask, mask)
                N=selected_pts_mask.sum()
                # NEW — this is per selected point
                selected_spawn_counts = torch.ones(N, device="cuda") * num_of_spawn
                #print(f"Number of points selected for spawning: {N.item()}")
                stds = training_args.std_scale*self.get_scaling[selected_pts_mask].repeat(num_of_spawn,1)
                means =torch.zeros((stds.size(0), 3),device="cuda")
                samples = torch.normal(mean=means, std=stds)
                rots = build_rotation(self.get_rotation[selected_pts_mask]).repeat(num_of_spawn,1,1)
                self._added_xyz = (torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(num_of_spawn, 1)).detach().requires_grad_(True)
                
                # self._added_scaling = self.scaling_inverse_activation(torch.tensor([0.1,0.1,0.1],device='cuda').repeat(N*num_of_spawn, 1)).detach().requires_grad_(True)
                self._added_rotation = torch.tensor([1.,0.,0.,0.],device='cuda').repeat(N*num_of_spawn, 1).detach().requires_grad_(True)
                # self._added_features_dc = ((torch.ones_like(self.get_features[:,0:1,:][selected_pts_mask])/2).repeat(num_of_spawn,1,1)).detach().requires_grad_(True)
                # self._added_features_rest = ((torch.zeros_like(self.get_features[:,1:,:][selected_pts_mask])).repeat(num_of_spawn,1,1)).detach().requires_grad_(True)
                self._added_opacity = self.inverse_opacity_activation(torch.tensor([0.1],device='cuda')).repeat(N*num_of_spawn, 1).detach().requires_grad_(True)
                
                self._added_scaling = (self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(num_of_spawn,1) / (0.8*num_of_spawn))).detach().requires_grad_(True)
                # self._added_rotation = (self.get_rotation[selected_pts_mask].repeat(num_of_spawn,1)).detach().requires_grad_(True)
                self._added_features_dc = (self.get_features[:,0:1,:][selected_pts_mask].repeat(num_of_spawn,1,1)).detach().requires_grad_(True)
                self._added_features_rest = (self.get_features[:,1:,:][selected_pts_mask].repeat(num_of_spawn,1,1)).detach().requires_grad_(True)
                # self._added_opacity = (self._opacity[selected_pts_mask].repeat(num_of_spawn,1)).detach().requires_grad_(True)


            ##### New Spawning code dependent on color distance #####
            elif args.new_spawn and args.grad_spawn:
                selected_pts_mask, selected_spawn_counts, old_xyz = self.spawn_points_gradient_based(grads, training_args, mask=mask, max_spawn_count=args.max_spawn_count)
            elif args.new_spawn and not args.grad_spawn and args.dyn:
                # Spawn dyn from color distance  - works
                #selected_pts_mask, selected_spawn_counts, old_xyz = self.spawn_points_color_distance_function(gt_image, image, i, viewpoint_cam, visibility_filter, name_of_exp, training_args, args)
                
                # spawn constant from distance - works
                #selected_pts_mask, selected_spawn_counts, old_xyz = self.spawn_constant_from_distance(gt_image, image, viewpoint_cam, i, name_of_exp, training_args, args)
                
                # Spawn dyn from mutliview:
                selected_pts_mask, selected_spawn_counts, old_xyz = self.spawn_points_color_distance_multiview(
                                    gt_image, image, viewpoint_cam, i, visibility_filter, name_of_exp, training_args, args)


            # elif args.new_spawn and args.accumulated_spawn:
            #     selected_pts_mask, selected_spawn_counts, old_xyz = self.spawn_gaussians_by_error(self, gt_image, image, viewpoint_cam, training_args, args, i, name_of_exp, visibility_filter)
                   
            ##### End of it 
            #####

            if args.col_mask:
                random_idx = torch.randint(0, len(gt_image), (1,))
                selected_pts_mask, selected_spawn_counts, old_xyz = self.extract_errorful_gaussians_from_diff_map(gt_image[random_idx], image[random_idx], viewpoint_cam[random_idx], args)


            
            # choose 5 top points that are not close to each other
            def select_topk_spatially_diverse_points(xyz: torch.Tensor, spawn_counts: torch.Tensor, k: int = 5, min_dist: float = 0.05):
                """
                Selects top-k points with the highest spawn counts that are not too close to each other.

                Args:
                    xyz (Tensor): [N, 3] positions of selected Gaussians
                    spawn_counts (Tensor): [N] number of Gaussians to spawn per point
                    k (int): number of points to select
                    min_dist (float): minimum allowed Euclidean distance between points

                Returns:
                    Tensor: [M <= k] xyz positions of selected points
                """
                # Sort by descending spawn_counts
                sorted_indices = torch.argsort(spawn_counts, descending=True)
                selected_xyz = []
                
                for idx in sorted_indices:
                    candidate = xyz[idx]
                    too_close = False
                    for selected in selected_xyz:
                        if torch.norm(candidate - selected) < min_dist:
                            too_close = True
                            break
                    if not too_close:
                        selected_xyz.append(candidate)
                    if len(selected_xyz) == k:
                        break

                if selected_xyz:
                    return torch.stack(selected_xyz)
                else:
                    return torch.empty((0, 3), device=xyz.device)

            # === Usage ===
            if not args.is_cluster:
                selected_xyz = old_xyz[selected_pts_mask]
                #selected_spawn_counts = spawn_counts[selected_pts_mask].repeat(num_of_spawn)

                topk_xyz = select_topk_spatially_diverse_points(selected_xyz, selected_spawn_counts, min_dist=args.min_dist, k=5)
                self.max_gaussian_xyzs = topk_xyz.detach().clone().requires_grad_(True)
        
        elif training_args.spawn_type=='random':
        # Spawn
            num_of_spawn=training_args.num_of_spawn
            selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= training_args.densify_grad_threshold, True, False)
            selected_pts_mask = torch.logical_and(selected_pts_mask, mask)
            N=selected_pts_mask.sum()

            self._added_xyz = (torch.rand([N*num_of_spawn,3],device='cuda')*(self._xyz_bound_max-self._xyz_bound_min)+self._xyz_bound_min).detach().requires_grad_(True)
            
            # self._added_scaling = self.scaling_inverse_activation(torch.tensor([0.1,0.1,0.1],device='cuda').repeat(N*num_of_spawn, 1)).detach().requires_grad_(True)
            self._added_rotation = torch.tensor([1.,0.,0.,0.],device='cuda').repeat(N*num_of_spawn, 1).detach().requires_grad_(True)
            # self._added_features_dc = ((torch.ones_like(self.get_features[:,0:1,:][selected_pts_mask])/2).repeat(num_of_spawn,1,1)).detach().requires_grad_(True)
            # self._added_features_rest = ((torch.zeros_like(self.get_features[:,1:,:][selected_pts_mask])).repeat(num_of_spawn,1,1)).detach().requires_grad_(True)
            self._added_opacity = self.inverse_opacity_activation(torch.tensor([0.1],device='cuda')).repeat(N*num_of_spawn, 1).detach().requires_grad_(True)
            
            self._added_scaling = (self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(num_of_spawn,1) / (0.8*num_of_spawn))).detach().requires_grad_(True)
            # self._added_rotation = (self.get_rotation[selected_pts_mask].repeat(num_of_spawn,1)).detach().requires_grad_(True)
            self._added_features_dc = (self.get_features[:,0:1,:][selected_pts_mask].repeat(num_of_spawn,1,1)).detach().requires_grad_(True)
            self._added_features_rest = (self.get_features[:,1:,:][selected_pts_mask].repeat(num_of_spawn,1,1)).detach().requires_grad_(True)
            # self._added_opacity = (self._opacity[selected_pts_mask].repeat(num_of_spawn,1)).detach().requires_grad_(True)            
        # Optimizer

        l = [
            {'params': [self._added_xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "added_xyz"},
            {'params': [self._added_features_dc], 'lr': training_args.feature_lr, "name": "added_f_dc"},
            {'params': [self._added_features_rest], 'lr': training_args.feature_lr / 20.0, "name": "added_f_rest"},
            {'params': [self._added_opacity], 'lr': training_args.opacity_lr, "name": "added_opacity"},
            {'params': [self._added_scaling], 'lr': training_args.scaling_lr, "name": "added_scaling"},
            {'params': [self._added_rotation], 'lr': training_args.rotation_lr, "name": "added_rotation"}
        ]

        # l = [
        #     {'params': [self._added_xyz], 'lr': 0.00016, "name": "added_xyz"},
        #     {'params': [self._added_features_dc], 'lr': 0.0025, "name": "added_f_dc"},
        #     {'params': [self._added_features_rest], 'lr': 0.0025/20, "name": "added_f_rest"},
        #     {'params': [self._added_opacity], 'lr': 0.005, "name": "added_opacity"},  # <== Critical fix
        #     {'params': [self._added_scaling], 'lr': 0.005, "name": "added_scaling"},
        #     {'params': [self._added_rotation], 'lr': 0.001, "name": "added_rotation"}
        # ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.color_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
               
        added_mask=torch.zeros((self.get_xyz.shape[0]), device="cuda", dtype=torch.bool)
        added_mask[-self._added_xyz.shape[0]:]=True
        print(f"Shape of added mask: {added_mask.shape}, number of added points: {added_mask.sum().item()}")
        self._added_mask=added_mask
        
        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        #print(f"xyz_grad_accum: {self.xyz_gradient_accum[update_filter].mean()}")
        #print(f"Are there any nans in xyz_grad_accum: {torch.sum(torch.isnan(self.xyz_gradient_accum))}")
        self.color_gradient_accum[update_filter] += torch.norm(self._features_dc.grad[update_filter].squeeze(), dim=-1, keepdim=True)
        self.denom[update_filter] += 1
    
    def query_ntc(self):
        mask, self._d_xyz, self._d_rot = self.ntc(self._xyz)
        
        self._new_xyz = self._d_xyz + self._xyz
        self._new_rot = self.rotation_compose(self._rotation, self._d_rot)
        if self._rotate_sh == True:
            self._new_feature = torch.cat((self._features_dc, self._features_rest), dim=1) # [N, SHs, RGB]
                    
            # self._d_rot_matrix=build_rotation(self._d_rot)
            # self._new_feature[mask][:,1:4,0] = rotate_sh_by_matrix(self._features_rest[mask][...,0],1,self._d_rot_matrix[mask])
            # self._new_feature[mask][:,1:4,1] = rotate_sh_by_matrix(self._features_rest[mask][...,1],1,self._d_rot_matrix[mask])
            # self._new_feature[mask][:,1:4,2] = rotate_sh_by_matrix(self._features_rest[mask][...,2],1,self._d_rot_matrix[mask])
            
            # This is a bit faster...      
            permuted_feature = self._new_feature.permute(0, 2, 1)[mask] # [N, RGB, SHs]
            reshaped_feature = permuted_feature.reshape(-1,4)
            repeated_quat = self.rotation_activation(self._d_rot[mask]).repeat(3, 1)
            rotated_reshaped_feature = rotate_sh_by_quaternion(sh=reshaped_feature[...,1:],l=1,q=repeated_quat) # [3N, SHs(l=1)]
            rotated_permuted_feature = rotated_reshaped_feature.reshape(-1,3,3) # [N, RGB, SHs(l=1)]
            self._new_feature[mask][:,1:4]=rotated_permuted_feature.permute(0,2,1)  



    def update_by_ntc(self):
        self._xyz = self.get_xyz.detach()
        self._features_dc = self.get_features[:,0:1,:].detach()
        self._features_rest = self.get_features[:,1:,:].detach()
        self._opacity = self._opacity.detach()
        self._scaling = self._scaling.detach()
        self._rotation = self.get_rotation.detach()
        
        self._d_xyz = None
        self._d_rot = None
        self._d_rot_matrix = None
        self._d_scaling = None
        self._d_opacity = None
        
        self._new_xyz = None
        self._new_rot = None
        self._new_scaling = None
        self._new_opacity = None
        self._new_feature = None
                    
    def get_contracted_xyz(self):
        with torch.no_grad():
            xyz = self.get_xyz
            xyz_bound_min, xyz_bound_max = self.get_xyz_bound(86.6)
            normalzied_xyz=(xyz-xyz_bound_min)/(xyz_bound_max-xyz_bound_min)
            return normalzied_xyz
    
    def get_xyz_bound(self, percentile=86.6):
        with torch.no_grad():
            if self._xyz_bound_min is None:
                half_percentile = (100 - percentile) / 200
                self._xyz_bound_min = torch.quantile(self._xyz,half_percentile,dim=0)
                self._xyz_bound_max = torch.quantile(self._xyz,1 - half_percentile,dim=0)
            return self._xyz_bound_min, self._xyz_bound_max

    def training_one_frame_setup(self,training_args):
        ntc_conf_path=training_args.ntc_conf_path
        with open(ntc_conf_path) as ntc_conf_file:
            ntc_conf = ctjs.load(ntc_conf_file)
        if training_args.only_mlp:
            model=tcnn.Network(n_input_dims=3, n_output_dims=8, network_config=ntc_conf["network"]).to(torch.device("cuda"))
        else:
            model=tcnn.NetworkWithInputEncoding(n_input_dims=3, n_output_dims=8, encoding_config=ntc_conf["encoding"], network_config=ntc_conf["network"]).to(torch.device("cuda"))
        self.ntc=NeuralTransformationCache(model,self.get_xyz_bound()[0],self.get_xyz_bound()[1])
        self.ntc.load_state_dict(torch.load(training_args.ntc_path))
        self._xyz_bound_min = self.ntc.xyz_bound_min
        self._xyz_bound_max = self.ntc.xyz_bound_max
        if training_args.ntc_lr is not None:
            ntc_lr=training_args.ntc_lr
        else:
            ntc_lr=ntc_conf["optimizer"]["learning_rate"]
        self.ntc_optimizer = torch.optim.Adam(self.ntc.parameters(),
                                                lr=ntc_lr)            
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.color_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        
    def get_masked_gaussian(self, mask):        
        new_gaussian = GaussianModel(self.max_sh_degree)
        new_gaussian._xyz = self.get_xyz[mask].detach()
        new_gaussian._features_dc = self.get_features[:,0:1,:][mask].detach()
        new_gaussian._features_rest = self.get_features[:,1:,:][mask].detach()
        new_gaussian._scaling = self.scaling_inverse_activation(self.get_scaling)[mask].detach()
        new_gaussian._rotation = self.get_rotation[mask].detach()
        new_gaussian._opacity = self.inverse_opacity_activation(self.get_opacity)[mask].detach()
        new_gaussian.xyz_gradient_accum = torch.zeros((new_gaussian._xyz.shape[0], 1), device="cuda")
        new_gaussian.color_gradient_accum = torch.zeros((new_gaussian._xyz.shape[0], 1), device="cuda")
        new_gaussian.denom = torch.zeros((new_gaussian._xyz.shape[0], 1), device="cuda")
        new_gaussian.max_radii2D = torch.zeros((new_gaussian._xyz.shape[0]), device="cuda")
        return new_gaussian
    
    def query_ntc_eval(self):
        with torch.no_grad():
            mask, self._d_xyz, self._d_rot = self.ntc(self.get_xyz)
            
            self._new_xyz = self._d_xyz + self._xyz
            self._new_rot = self.rotation_compose(self._rotation, self._d_rot)
            if self._rotate_sh == True:
                self._new_feature = torch.cat((self._features_dc, self._features_rest), dim=1) # [N, SHs, RGB]
                # This is a bit faster...      
                permuted_feature = self._new_feature.permute(0, 2, 1)[mask] # [N, RGB, SHs]
                reshaped_feature = permuted_feature.reshape(-1,4)
                repeated_quat = self.rotation_activation(self._d_rot[mask]).repeat(3, 1)
                rotated_reshaped_feature = rotate_sh_by_quaternion(sh=reshaped_feature[...,1:],l=1,q=repeated_quat) # [3N, SHs(l=1)]
                rotated_permuted_feature = rotated_reshaped_feature.reshape(-1,3,3) # [N, RGB, SHs(l=1)]
                self._new_feature[mask][:,1:4]=rotated_permuted_feature.permute(0,2,1)  