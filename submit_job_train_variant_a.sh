#!/bin/bash
#SBATCH --job-name=variantA-fpn-dcae
#SBATCH --output=slurm_variant_a_fpn_dcae_%j.out
#SBATCH --error=slurm_variant_a_fpn_dcae_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=72:00:00

set -eo pipefail

echo "Job running on node: $SLURM_JOB_NODELIST"
echo "Starting Variant A ACT training with FPN feature fusion + DCAE-style decoder..."

source ~/.bashrc
conda activate python311t
set -u

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TASK_NAME="sim_transfer_cube_scripted"
TARGET_CAMERA="top"
CKPT_DIR="./checkpoints/checkpoints_variant_a_${TASK_NAME}_${TARGET_CAMERA}_fpn_dcae_accum16_focus_masked_region"

mkdir -p "${CKPT_DIR}"
echo "Checkpoint dir: ${CKPT_DIR}"
echo "Task: ${TASK_NAME}, target camera: ${TARGET_CAMERA}"
nvidia-smi
python -u imitate_episodes_variant_a.py \
    --task_name "${TASK_NAME}" \
    --ckpt_dir "${CKPT_DIR}" \
    --policy_class ACT \
    --target_camera "${TARGET_CAMERA}" \
    --kl_weight 10 \
    --chunk_size 100 \
    --hidden_dim 512 \
    --batch_size 8 \
    --grad_accum_steps 2 \
    --dim_feedforward 3200 \
    --lr 2e-5 \
    --seed 10 \
    --num_epochs 8000 \
    --lambda_roi 1.0 \
    --lambda_recon_grad 0.25 \
    --lambda_sem 0.1 \
    --lambda_sig 0.0 \
    --roi_background_weight 1.0 \
    --roi_detail_weight 10.0 \
    --comm_num_queries 8 \
    --comm_layers 2 \
    --comm_detach_warmup 0 \
    --ema_momentum 0.99 \
    --focus_masked_region
