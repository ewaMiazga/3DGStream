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
import time
import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, quaternion_loss, d_xyz_gt, d_rot_gt
from gaussian_renderer import render, network_gui
import sys
import json
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from utils.debug_utils import save_tensor_img
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import re
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

masked_i = 0
name_of_exp = []
import matplotlib.pyplot as plt

def render_and_save_gradients(viewpoint_cam, gaussians, i, pipe, background = torch.tensor([0,0,0], dtype=torch.float32, device="cuda")):

    # === Step 1: Create a shallow Gaussian subset based on the mask ===
    
    render_pkg = render(viewpoint_cam, gaussians, pipe, background)
    image, depth, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["depth"],render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

    from torchvision.utils import save_image

    # Assuming `image` is [3, H, W] and values in [0, 1]
    save_image(image, f'output_dist_maps/{i}_rendered_image_grad.png')


def get_all_viewpoints(dataset, scene, opt, pipe, background, gaussians):
    """
    Get all viewpoints from the dataset and render them using the provided pipeline.
    """
    # Prepare lists to store all outputs
    rendered_images = []
    depth_maps = []
    ground_truth_images = []
    camera_list = []
    visibility_filters = []
    viewspace_points_list = []
    radii_list = []

    # Get all train cameras
    viewpoint_cams = scene.getTrainCameras()
    viewpoint_stack = viewpoint_cams.copy()  # Copy to preserve the original list

    # Loop through all available viewpoints
    for cam in viewpoint_stack:
        # Render current view
        render_pkg = render(cam, gaussians, pipe, background)

        # Extract rendered outputs
        image = render_pkg["render"]
        depth = render_pkg["depth"]
        viewspace_points = render_pkg["viewspace_points"]
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]

        # Get ground truth image for this camera (if available)
        gt_img = cam.original_image.cuda()

        # Append everything to lists
        rendered_images.append(image)
        depth_maps.append(depth)
        viewspace_points_list.append(viewspace_points)
        visibility_filters.append(visibility_filter)
        radii_list.append(radii)
        ground_truth_images.append(gt_img)
        camera_list.append(cam)

    return rendered_images, depth_maps, ground_truth_images, camera_list, visibility_filters, viewspace_points_list, radii_list


def training_one_frame(dataset, opt, pipe, load_iteration, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, args):
    global masked_i
    masked_i += 1
    start_time=time.time()
    last_s1_res = []
    last_s2_res = []
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree,opt.rotate_sh)
    scene = Scene(dataset, gaussians, load_iteration=load_iteration, shuffle=False)
    gaussians.training_one_frame_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    s1_start_time=time.time()
    # Train the NTC
    for iteration in range(first_iter, opt.iterations + 1):        
        iter_start.record()

        # gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()
                     
        # Query the NTC
        #print("Querying NTC at iteration {}".format(iteration))
        #print("XYZ shape:", gaussians._xyz.shape)
        #print("Gaussians fearure dc shape: ", gaussians._features_dc.shape)
        gaussians.query_ntc()
        
        loss = torch.tensor(0.).cuda()
        
        
        # A simple 
        for batch_iteraion in range(opt.batch_size):
        
            # Pick a random Camera
            if not viewpoint_stack:
                viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
            
            # Render
            if (iteration - 1) == debug_from:
                pipe.debug = True
            render_pkg = render(viewpoint_cam, gaussians, pipe, background)
            image, depth, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["depth"],render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

            # Loss
            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 = l1_loss(image, gt_image)
            Lds = torch.tensor(0.).cuda()
            loss += (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
            
        loss/=opt.batch_size
        loss.backward()
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            s1_res = training_report(tb_writer, iteration, Ll1, Lds, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if s1_res is not None:
                last_s1_res.append(s1_res)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration=iteration, save_type='all')

            # Tracking Densification Stats
            if iteration > opt.densify_from_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                #print("Shape of visibility filter: ", visibility_filter.shape)
                #print("View space point tensor shape: ", viewspace_point_tensor.shape)
                #print("Is full of nans: ", torch.isnan(viewspace_point_tensor).any())

                ### Now we add the densification stats and xyz_gradient_accum.mean != 0
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.ntc_optimizer.step()
                gaussians.ntc_optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.output_path + "/chkpnt" + str(iteration) + ".pth")

    s1_end_time=time.time()
    # Densify
    if(opt.iterations_s2>0):
    # Dump the NTC
        scene.dump_NTC()
    # Update Gaussians by NTC
        gaussians.update_by_ntc()
    # Prune, Clone and setting up  

        ### Idea to check where the gaussians with high space-view are in the rendered image
        # grads = gaussians.xyz_gradient_accum / gaussians.denom
        # grads[grads.isnan()] = 0.0
        # grads_norms = torch.norm(grads, dim=-1)
        # high_grad_mask = grads_norms >= 0.00007
        # masked_gaus = gaussians.clone_subset(high_grad_mask)
        # # reneder and save gradients
        # render_and_save_gradients(viewpoint_cam, masked_gaus, masked_i, pipe, background)

        #print(f"Mean of xyz gaussians before setup: {gaussians.xyz_gradient_accum.mean(dim=0)}") ### != 0 
        rendered_images, depth_maps, ground_truth_images, camera_list, visibility_filters, viewspace_points_list, radii_list = get_all_viewpoints(dataset, scene, opt, pipe, background, gaussians)
        gaussians.training_one_frame_s2_setup(ground_truth_images, rendered_images, masked_i, camera_list, visibility_filters, name_of_exp, pipe, opt, args)
        #print(f"Mean of xyz gaussians after setup: {gaussians.xyz_gradient_accum.mean(dim=0)}")
        progress_bar = tqdm(range(opt.iterations, opt.iterations + opt.iterations_s2), desc="Training progress of Stage 2")    
    

    # Train the new Gaussians
    for iteration in range(opt.iterations + 1, opt.iterations + opt.iterations_s2 + 1):        
        iter_start.record()
                     
        # Update Learning Rate
        # gaussians.update_learning_rate(iteration)
        
        loss = torch.tensor(0.).cuda()
        
        for batch_iteraion in range(opt.batch_size):
        
            # Pick a random Camera
            if not viewpoint_stack:
                viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
            
            # # Render OLD
            if (iteration - 1) == debug_from:
                    pipe.debug = True
            if not args.new_loss:
                render_pkg = render(viewpoint_cam, gaussians, pipe, background)
                image, depth, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["depth"],render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
                # Loss
                gt_image = viewpoint_cam.original_image.cuda()
                Ll1 = l1_loss(image, gt_image)
                loss += (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
            


            ### WORKING RENDERING PATCHES
            # Render New - localize the gaussian with the highest grad change and use only the patch around it 
            # grads: [N, 3] — assume already computed

            else: 
                render_pkg = render(viewpoint_cam, gaussians, pipe, background)
                image, depth, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["depth"],render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
                # Loss
                gt_image = viewpoint_cam.original_image.cuda()

                #max_gaussian_xyz = gaussians.max_gaussian_xyz
                max_gaussian_xyzs = gaussians.max_gaussian_xyzs

                def project_xyz_to_screen(xyz, camera):
                    # Homogeneous coordinates
                    N = xyz.shape[0]
                    xyz_h = torch.cat([xyz, torch.ones((N, 1), device=xyz.device)], dim=1)  # [4]
                    screen_coord = (camera.full_proj_transform.T @ xyz_h.T).T # [4]
                    screen_coord = screen_coord.clone()
                    screen_coord = screen_coord / screen_coord[:, 3:4].clone()   # Perspective divide
                    x_ndc, y_ndc = screen_coord[:, 0], screen_coord[:, 1]  # [-1, 1]

                    # Convert to pixel coords
                    W, H = camera.image_width, camera.image_height
                    x = ((x_ndc + 1) / 2 * W).long().clamp(0, W-1)
                    y = ((y_ndc + 1) / 2 * H).long().clamp(0, H-1)
                    return x, y
                
                x_pixels, y_pixels = project_xyz_to_screen(max_gaussian_xyzs, viewpoint_cam)

                patch_size = 128  # or any small region
                W, H = viewpoint_cam.image_width, viewpoint_cam.image_height
                # x_min, x_max = max(0, x_pixel - patch_size//2), min(W, x_pixel + patch_size//2)
                # y_min, y_max = max(0, y_pixel - patch_size//2), min(H, y_pixel + patch_size//2)

                # attention_mask = torch.zeros((H, W), dtype=torch.bool, device="cuda")
                # attention_mask[y_min:y_max, x_min:x_max] = True

                attention_mask = torch.zeros((H, W), dtype=torch.bool, device="cuda")
                for x_pixel, y_pixel in zip(x_pixels, y_pixels):
                    x_min, x_max = max(0, x_pixel - patch_size // 2), min(W, x_pixel + patch_size // 2)
                    y_min, y_max = max(0, y_pixel - patch_size // 2), min(H, y_pixel + patch_size // 2)
                    attention_mask[y_min:y_max, x_min:x_max] = True

                # Expand to 3-channel mask
                mask3 = attention_mask.unsqueeze(0).expand_as(image)

                masked_render = image * mask3
                masked_gt = gt_image * mask3

                valid_pixels = attention_mask.sum()
                if valid_pixels > 0:
                    Ll1 = torch.abs(masked_render - masked_gt).sum() / valid_pixels
                else:
                    Ll1 = torch.tensor(0.0, device=image.device)

                # save masked render and masked ground truth
                os.makedirs(f'{name_of_exp[0]}/masked', exist_ok = True)
                save_tensor_img(masked_render, os.path.join(f'{name_of_exp[0]}/masked',f'{masked_i}_masked_rendering2'))
                save_tensor_img(masked_gt, os.path.join(f'{name_of_exp[0]}/masked',f'{masked_i}_masked_ground_truth2'))

                #Ll1 = l1_loss(masked_render, masked_gt)
                # local patch loss calculated 
                ##### CHANGE TO HAVE EQUAL RATIO OF L1 AND SSIM
                loss += (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(masked_render, masked_gt))
                #loss += (Ll1 + opt.lambda_dssim * (1.0 - ssim(masked_render, masked_gt)))

                # global loss calculated
                Ll1 = l1_loss(image, gt_image)
                loss += (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

        
        loss/=opt.batch_size
        loss.backward()
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if (iteration - opt.iterations) % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations + opt.iterations_s2:
                progress_bar.close()

            # Log and save
            s2_res = training_report(tb_writer, iteration, Ll1, Lds, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if s2_res is not None:
                last_s2_res.append(s2_res)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration=iteration, save_type='added')
                      
            # Densification
            if (iteration - opt.iterations) % opt.densification_interval == 0:
                #print(f"Mean of xyz gaussians before adding and pruning: {gaussians.xyz_gradient_accum.mean(dim=0)}")
                #print(f"Adding and pruning gaussians at iteration {iteration}")
                gaussians.adding_and_prune(opt,scene.cameras_extent)
                             
            # Optimizer step
            if iteration < opt.iterations + opt.iterations_s2:
                # Check if colors are updated
                #print("Color before optimizer step: ", gaussians._added_features_dc.mean(dim=0))
                #print("Rest Color before optmizer step: ", gaussians._added_features_rest.mean(dim=0))
                gaussians.optimizer.step()
                #print("Color after optimizer step: ", gaussians._added_features_dc.mean(dim=0))
                #print("Rest Color after optmizer step: ", gaussians._added_features_rest.mean(dim=0))
                gaussians.optimizer.zero_grad(set_to_none = True)
    s2_end_time=time.time()
    
    # 计算总训练时间 - calculate total training time 
    pre_time = s1_start_time - start_time
    s1_time = s1_end_time - s1_start_time
    s2_time = s2_end_time - s1_end_time
           
    return last_s1_res, last_s2_res, pre_time, s1_time, s2_time

def prepare_output_and_logger(args):    
    if not args.output_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.output_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.output_path))
    os.makedirs(args.output_path, exist_ok = True)
    with open(os.path.join(args.output_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.output_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, Lds, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    last_test_psnr=0.0
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/ds_loss', Lds.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                            #   {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]}
                              )

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    # if scene.gaussians._added_mask is not None:
                    #     added_pkg = renderFunc(viewpoint, scene.gaussians.get_masked_gaussian(scene.gaussians._added_mask), *renderArgs)
                    image, depth = torch.clamp(render_pkg["render"], 0.0, 1.0), render_pkg["depth"]
                    depth_vis=depth/(depth.max()+1e-5)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_image(config['name'] + "_view_{}/render".format(viewpoint.image_name), image, global_step=iteration)
                        # tb_writer.add_image(config['name'] + "_view_{}/diff".format(viewpoint.image_name), (gt_image-image).abs().mean(dim=0, keepdim=True), global_step=iteration)
                        # tb_writer.add_image(config['name'] + "_view_{}/depth".format(viewpoint.image_name), depth_vis, global_step=iteration)
                        # if scene.gaussians._added_mask is not None:
                        #     tb_writer.add_image(config['name'] + "_view_{}/added_gaussians".format(viewpoint.image_name), torch.clamp(added_pkg["render"], 0.0, 1.0), global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_image(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image, global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                if config['name'] == 'test':
                    last_test_psnr = psnr_test
                    last_test_image = image
                    last_gt = gt_image

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()
        
        return {#'last_test_psnr':last_test_psnr.cpu().numpy()
                  'last_test_psnr':float(last_test_psnr)
                , 'last_test_image':last_test_image.cpu()
                , 'last_points_num':scene.gaussians.get_xyz.shape[0]
                # , 'last_gt':last_gt.cpu()
                }

def train_one_frame(lp,op,pp,args):
    args.save_iterations.append(args.iterations + args.iterations_s2)
    if args.depth_smooth==0:
        args.bwd_depth=False
    print("Optimizing " + args.output_path)
    res_dict={}
    if(args.opt_type=='3DGStream'):
        s1_ress, s2_ress, pre_time, s1_time, s2_time = training_one_frame(lp.extract(args), op.extract(args), pp.extract(args), args.load_iteration, args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args)

        # All done
        print("\nTraining complete.")
        print(f"Preparation: {pre_time}")
        if pre_time > 2:
            print(f"If preparation is time-consuming, consider down-scaling the images BEFORE running 3DGStream.")
        print(f"Stage 1: {s1_time}")
        print(f"Stage 2: {s2_time}")
        if s1_ress !=[]:
            for idx, s1_res in enumerate(s1_ress):
                save_tensor_img(s1_res['last_test_image'],os.path.join(args.output_path,f'{idx}_rendering1'))
                res_dict[f'stage1/psnr_{idx}']=s1_res['last_test_psnr']
                res_dict[f'stage1/points_num_{idx}']=s1_res['last_points_num']
            res_dict[f'stage1/time']=s1_time
        if s2_ress !=[]:
            for idx, s2_res in enumerate(s2_ress):
                save_tensor_img(s2_res['last_test_image'],os.path.join(args.output_path,f'{idx}_rendering2'))
                res_dict[f'stage2/psnr_{idx}']=s2_res['last_test_psnr']
                res_dict[f'stage2/points_num_{idx}']=s2_res['last_points_num']
            res_dict[f'stage2/time']=s2_time
    return res_dict 

def train_frames(lp, op, pp, args):
    # Initialize system state (RNG)
    safe_state(args.quiet)
    video_path=args.video_path
    output_path=args.output_path
    model_path=args.model_path
    load_iteration = args.load_iteration
    sub_paths = os.listdir(video_path)
    pattern = re.compile(r'frame(\d+)')
    frames = sorted(
        (item for item in sub_paths if pattern.match(item)),
        key=lambda x: int(pattern.match(x).group(1))
    )
    frames=frames[args.frame_start-1:args.frame_end]
    if args.frame_start==1:
        args.load_iteration = args.first_load_iteration
    
    ### Add for overall stats
    global_psnr_sum = 0.0
    global_l1_sum = 0.0
    total_test_views = 0

    for frame in frames:
        start_time = time.time()
        args.source_path = os.path.join(video_path, frame)
        args.output_path = os.path.join(output_path, frame)
        args.model_path = model_path
        ##train_one_frame(lp,op,pp,args)
        res = train_one_frame(lp, op, pp, args)
        if 'stage2/psnr_0' in res:
            global_psnr_sum += res['stage2/psnr_0']
            total_test_views += 1
            print(f"[GLOBAL] Accumulated PSNR: {global_psnr_sum:.3f} over {total_test_views} frames")


        print(f"Frame {frame} finished in {time.time()-start_time} seconds.")
        model_path = args.output_path
        args.load_iteration = load_iteration
        torch.cuda.empty_cache()
    
    if total_test_views > 0:
        avg_psnr = global_psnr_sum / total_test_views
        print(f"[FINAL] Average PSNR across all frames: {avg_psnr:.3f}")

import os
import imageio
from glob import glob
from natsort import natsorted

def create_video_from_frames(frame_dirs, image_filename, output_path, fps=10):
    """
    Given a list of frame directories, compile a video from image_filename.
    """
    images = []
    for frame_dir in tqdm(frame_dirs, desc=f"Creating {os.path.basename(output_path)}"):
        image_path = os.path.join(frame_dir, image_filename)
        if os.path.isfile(image_path):
            try:
                img = imageio.imread(image_path)
                images.append(img)
            except Exception as e:
                print(f"⚠️ Skipping {image_path}: {e}")
        else:
            print(f"⚠️ Missing: {image_path}")

    if images:
        imageio.mimsave(output_path, images, fps=fps)
        print(f"✅ Saved: {output_path}")
    else:
        print(f"❌ No valid images found for: {output_path}")

def process_all_experiments(root_dir, fps=10):
    """
    Recursively search for experiments under root_dir and create videos.
    """
    frame_dirs = natsorted(glob(os.path.join(root_dir, "frame*")))

    print(f"\n🎬 Processing: {root_dir}")

    for image_file, video_name in [
        ("0_rendering1.png", "rendering1_video.mp4"),
        ("0_rendering2.png", "rendering2_video.mp4"),
    ]:
        output_path = os.path.join(root_dir, video_name)
        create_video_from_frames(frame_dirs, image_file, output_path, fps=fps)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--frame_start', type=int, default=1)
    parser.add_argument('--frame_end', type=int, default=150)
    parser.add_argument('--load_iteration', type=int, default=None)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1, 50, 100])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[1, 50, 100])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--read_config", action='store_true', default=False)
    parser.add_argument("--config_path", type=str, default = None)
    parser.add_argument("--is_acc", action="store_true", help="If errors are accumulated")
    parser.add_argument("--max_spawn_count", type=int, default=1, help="Max number of spawn count")
    parser.add_argument("--min_dist", type=float, default=0.1, help="Minimum distance between masks")
    parser.add_argument("--is_cluster", action="store_true", help="If clustering is used")
    parser.add_argument("--cluster_eps", type=float, default=1.0, help="Epsilon for clustering")
    parser.add_argument("--min_samples", type=int, default=10, help="Minimum samples for clustering")
    parser.add_argument("--new_loss", action="store_true", help="If new loss is used")
    parser.add_argument("--new_spawn", action="store_true", help="If new spawn is used")
    parser.add_argument("--grad_spawn", action="store_true", help="If gradient spawn is used")
    parser.add_argument("--col_mask", action="store_true", help="If color mask is used")
    parser.add_argument("--accumulated_spawn", action="store_true", help="If accumulated spawn is used")
    parser.add_argument("--dyn", action="store_true", help="If dynamic spawn is used")
    print("Arguments: ", sys.argv[1:])
    args = parser.parse_args(sys.argv[1:])
    if args.output_path == "":
        args.output_path=args.model_path
    if args.read_config and args.config_path is not None:
        with open(args.config_path, 'r') as f:
            config = json.load(f)
        for key, value in config.items():
            if key not in ["output_path", "source_path", "model_path", "video_path", "debug_from"]:
                setattr(args, key, value)
    serializable_namespace = {k: v for k, v in vars(args).items() if isinstance(v, (int, float, str, bool, list, dict, tuple, type(None)))}
    json_namespace = json.dumps(serializable_namespace)
    os.makedirs(args.output_path, exist_ok = True)
    name_of_exp.append(args.output_path)
    with open(os.path.join(args.output_path, "cfg_args.json"), 'w') as f:
        f.write(json_namespace)
    # === Change this to your actual output path ===
    base_output_dir = args.output_path

    # train_one_frame(lp,op,pp,args)
    train_frames(lp,op,pp,args)


    # Create videos
    process_all_experiments(base_output_dir, fps=30)
    process_all_experiments(base_output_dir, fps=30)

