#!/bin/bash
#SBATCH --job-name=VariantB_Eval
#SBATCH --output=slurm_variant_b_eval_%j.out
#SBATCH --error=slurm_variant_b_eval_%j.err
#SBATCH --nodes=1                      # Run everything on one machine
#SBATCH --ntasks=1                     # One main task
#SBATCH --cpus-per-task=4              # 4 CPUs
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1                   # Request 1 GPU (Required for EGL)
#SBATCH --mem=64G                      # Request 64GB of System RAM
#SBATCH --time=10:00:00                # Max run time (hours)


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
python -u imitate_episodes_variant_b.py \
    --task_name sim_transfer_cube_scripted \
    --ckpt_dir /home/fe/xie/act_origin/checkpoints/checkpoints_variant_b_sim_transfer_cube_scripted_top_future_sim_three_stage_horizon_focus_masked_region \
    --policy_class ACT \
    --kl_weight 10 \
    --chunk_size 100 \
    --hidden_dim 512 \
    --batch_size 16 \
    --dim_feedforward 3200 \
    --lr 2e-5 \
    --seed 10 \
    --num_epochs 4000 \
    --target_camera top \
    --future_horizons 0 5 15 30 60 99\
    --future_image_height 240 \
    --future_image_width 320 \
    --future_layers 2 \
    --lambda_future_rgb 1.0 \
    --lambda_future_grad 0.25 \
    --lambda_future_latent 0.1 \
    --future_rgb_decay_alpha 0.03 \
    --future_latent_decay_alpha 0.01 \
    --temporal_agg \
    --eval
