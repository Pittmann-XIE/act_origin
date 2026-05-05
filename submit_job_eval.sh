#!/bin/bash
#SBATCH --job-name=BiA200
#SBATCH --output=slurm_eval_%j.out    # Standard output log (%j inserts job ID)
#SBATCH --error=slurm_eval_%j.err     # Standard error log
#SBATCH --nodes=1                      # Run everything on one machine
#SBATCH --ntasks=1                     # One main task
#SBATCH --cpus-per-task=4              # 4 CPUs
#SBATCH --gres=gpu:1                   # Request 1 GPU (Required for EGL)
#SBATCH --mem=64G                      # Request 64GB of System RAM
#SBATCH --time=72:00:00                # Max run time (hours)

# Print some basic info
echo "Job running on node: $SLURM_JOB_NODELIST"
echo "Starting evaluation with EGL rendering..."

# Load your environment 
source ~/.bashrc
conda activate python311t

# --- USE EGL BACKEND FOR HEADLESS GPU RENDERING ---
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
# --------------------------------------------------

# Run the evaluation script
python -u imitate_episodes_bisim_A.py \
    --task_name sim_transfer_cube_scripted \
    --ckpt_dir /home/fe/xie/act_origin/checkpoints/checkpoints_sim_transfer_cube_bisim_A_200\
    --policy_class ACT \
    --kl_weight 10 \
    --chunk_size 100 \
    --hidden_dim 512 \
    --batch_size 16 \
    --dim_feedforward 3200 \
    --lr 2e-5 \
    --seed 10 \
    --num_epochs 4000 \
    --temporal_agg \
    --eval