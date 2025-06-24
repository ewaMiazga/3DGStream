import os
import subprocess
from itertools import product

# Define parameter ranges
max_spawn_counts = [100]
if_accs = [False]
min_dists = [15]
is_clusters = [False]
cluster_eps_values = [1, 5, 10]
min_samples_values = [10, 50, 100]

base_command = [
    "python", "train_frames.py",
    "--read_config", "--config_path", "test/discussion/cfg_args.json",
    "-m", "test/discussion/discussion_init/",
    "-v", "discussion_scene",
    "--image", "images",
    "--first_load_iteration", "30000",
    "--iterations_s2", "300",
    "--quiet",
    "--new_loss", 
    "--new_spawn",
    "--dyn",
    #"--grad_spawn",
    "--col_mask",
    #"--accumulated_spawn"
]

total_runs = 0

multiple_runs = False  # Set to True if you want to run multiple configurations

if multiple_runs:

    for max_spawn_count, if_acc, min_dist, is_cluster in product(max_spawn_counts, if_accs, min_dists, is_clusters):
        
        # If clustering is not used
        if not is_cluster:
            run_name = f"spawn-{max_spawn_count}_acc-{if_acc}_minDist-{min_dist}_cluster-{is_cluster}"
            output_dir = f"output/Dimanche/FF/{run_name}"
            os.makedirs(output_dir, exist_ok=True)

            cmd = base_command + [
                "-o", output_dir,
                "--max_spawn_count", str(max_spawn_count),
                "--min_dist", str(min_dist)
            ]

            if if_acc:
                cmd.append("--is_acc")

            print(f"\n[Run {total_runs}] Running: {run_name}")
            with open(os.path.join(output_dir, "log.txt"), "w") as log_file:
                subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT)

            total_runs += 1

        # If clustering is used, iterate additional parameters
        else:
            for cluster_eps, min_samples in product(cluster_eps_values, min_samples_values):
                run_name = f"spawn-{max_spawn_count}_acc-{if_acc}_minDist-{min_dist}_cluster-{is_cluster}_eps-{cluster_eps}_samples-{min_samples}"
                output_dir = f"output/Dimanche/FF/{run_name}"
                os.makedirs(output_dir, exist_ok=True)

                cmd = base_command + [
                    "-o", output_dir,
                    "--max_spawn_count", str(max_spawn_count),
                    "--min_dist", str(min_dist),
                    "--cluster_eps", str(cluster_eps),
                    "--min_samples", str(min_samples)
                ]

                if if_acc:
                    cmd.append("--is_acc")
                if is_cluster:
                    cmd.append("--is_cluster")

                print(f"\n[Run {total_runs}] Running: {run_name}")
                with open(os.path.join(output_dir, "log.txt"), "w") as log_file:
                    subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT)

                total_runs += 1
    
else:
    #run_name = "spawn-100-acc-False_minDist-15_cluster-False"
    run_name = "0.2-spawn500-dyn-color-spawn-300iter-new-loss-color-map-min-dist-1-col-mask-undist"
    output_dir = f"output/Thursday/{run_name}"
    #output_dir = f"output/Monday/{run_name}"
    os.makedirs(output_dir, exist_ok=True)

    cmd = base_command + [
        "-o", output_dir,
        "--num_of_spawn", "500",
        "--max_spawn_count", "500",
        "--min_dist", "1",
        #"--densify_grad_threshold", "0.00007"
    ]
    print(f"\n[Run {total_runs}] Running: {run_name}")
    with open(os.path.join(output_dir, "log.txt"), "w") as log_file:
        subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT)
        
    total_runs += 1

print(f"\nAll done. Total runs: {total_runs}")