#!/bin/bash
#SBATCH --job-name=variantB-future-sim
#SBATCH --output=slurm_variant_b_future_%j.out
#SBATCH --error=slurm_variant_b_future_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=72:00:00

set -eo pipefail

echo "Job running on node: $SLURM_JOB_NODELIST"
echo "Starting Variant B ACT training with action-conditioned future visual simulator..."

source ~/.bashrc
conda activate python311t
set -u

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TASK_NAME="sim_transfer_cube_scripted"
TARGET_CAMERA="top"
CKPT_DIR="./checkpoints/checkpoints_variant_b_${TASK_NAME}_${TARGET_CAMERA}_future_sim_three_stage_horizon_focus_masked_region"

mkdir -p "${CKPT_DIR}"
echo "Checkpoint dir: ${CKPT_DIR}"
echo "Task: ${TASK_NAME}, target camera: ${TARGET_CAMERA}"
nvidia-smi
python -u imitate_episodes_variant_b.py \
    --task_name "${TASK_NAME}" \
    --ckpt_dir "${CKPT_DIR}" \
    --policy_class ACT \
    --target_camera "${TARGET_CAMERA}" \
    --kl_weight 10 \
    --chunk_size 100 \
    --hidden_dim 512 \
    --batch_size 2 \
    --grad_accum_steps 8 \
    --dim_feedforward 3200 \
    --lr 2e-5 \
    --seed 10 \
    --lambda_roi 1.0 \
    --lambda_recon_grad 0.25 \
    --lambda_sem 0.1 \
    --lambda_sig 0.0 \
    --lambda_future_rgb 1.0 \
    --lambda_future_grad 0.25 \
    --lambda_future_latent 0.1 \
    --future_horizons 0 5 15 30 60 99 \
    --future_image_height 240 \
    --future_image_width 320 \
    --future_layers 2 \
    --future_teacher_mix_steps 8000 \
    --future_rgb_decay_alpha 0.03 \
    --future_latent_decay_alpha 0.01 \
    --roi_background_weight 1.0 \
    --roi_detail_weight 10.0 \
    --comm_num_queries 8 \
    --comm_layers 2 \
    --comm_detach_warmup 0 \
    --ema_momentum 0.99 \
    --focus_masked_region \
    --three_stage \
    --stage1_epochs 4000 \
    --stage2_epochs 4000 \
    --stage3_epochs 4000 \
    --resume_ckpt "/home/fe/xie/act_origin/checkpoints/checkpoints_variant_b_sim_transfer_cube_scripted_top_future_sim_three_stage/policy_stage1_act_last_training.ckpt"
