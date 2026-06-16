#!/bin/bash
#SBATCH --job-name=VariantC_Eval
#SBATCH --output=slurm_variant_c_eval_%j.out
#SBATCH --error=slurm_variant_c_eval_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=10:00:00

set -eo pipefail

echo "Job running on node: $SLURM_JOB_NODELIST"
echo "Starting Variant C evaluation with EGL rendering..."

source ~/.bashrc
conda activate python311t
set -u

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

TASK_NAME="sim_transfer_cube_scripted"
TARGET_CAMERA="top"
# Used to find the checkpoint directory; model hyperparameters are loaded from
# ${CKPT_DIR}/training_config.json when available.
RQ_TOKENS=30
RQ_STAGES=4
RQ_CODEBOOK_BINS=512
CODEBOOK_DIM=128
CKPT_DIR="/home/fe/xie/act_origin/checkpoints/checkpoints_variant_c_${TASK_NAME}_${TARGET_CAMERA}_rq_N${RQ_TOKENS}_M${RQ_STAGES}_K${RQ_CODEBOOK_BINS}_D${CODEBOOK_DIM}_three_stage"

echo "Checkpoint dir: ${CKPT_DIR}"
echo "Task: ${TASK_NAME}, target camera: ${TARGET_CAMERA}"
echo "RQ enabled: true"
echo "RQ: tokens=${RQ_TOKENS}, stages=${RQ_STAGES}, codebook_bins=${RQ_CODEBOOK_BINS}, codebook_dim=${CODEBOOK_DIM}"
nvidia-smi

python -u imitate_episodes_variant_c.py \
    --task_name "${TASK_NAME}" \
    --ckpt_dir "${CKPT_DIR}" \
    --policy_class ACT \
    --kl_weight 10 \
    --chunk_size 100 \
    --hidden_dim 512 \
    --batch_size 16 \
    --dim_feedforward 3200 \
    --lr 2e-5 \
    --seed 10 \
    --num_epochs 4000 \
    --target_camera "${TARGET_CAMERA}" \
    --future_horizons 0 5 15 30 60 99 \
    --future_image_height 240 \
    --future_image_width 320 \
    --future_layers 2 \
    --lambda_future_rgb 1.0 \
    --lambda_future_grad 0.25 \
    --lambda_future_latent 0.1 \
    --future_rgb_decay_alpha 0.03 \
    --future_latent_decay_alpha 0.01 \
    --using_RQ \
    --codebook_dim "${CODEBOOK_DIM}" \
    --lambda_vq 1.0 \
    --lambda_vq_commit 0.25 \
    --vq_warmup_epochs 0 \
    --rq_num_tokens "${RQ_TOKENS}" \
    --rq_num_stages "${RQ_STAGES}" \
    --rq_codebook_bins "${RQ_CODEBOOK_BINS}" \
    --rq_dead_code_restart_max_fraction 0.05 \
    --temporal_agg \
    --using_RQ \
    --eval
