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
#SBATCH --time=5:00:00

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

# Pass the checkpoint directory to sbatch, for example:
#   sbatch submit_job_eval_variant_c.sh \
#     checkpoints/checkpoints_variant_c_sim_transfer_cube_scripted_top_rq_N30_M4_K512_D128_2_three_stage
CKPT_DIR="${1:-checkpoints/checkpoints_variant_c_sim_transfer_cube_scripted_top_split_vq_N30_C8_K512_D512_three_stage}"
CKPT_NAME="$(basename "${CKPT_DIR}")"

# Reconstruct the quantizer architecture encoded in the checkpoint name.
# "hierarchy" RQ checkpoints use the same RQ module; the hierarchy comes from
# using progressively more residual stages during training/evaluation.
if [[ "${CKPT_NAME}" =~ _split_vq_N([0-9]+)_C([0-9]+)_K([0-9]+)_D([0-9]+) ]]; then
    CODEBOOK_TYPE="split_vq"
    RQ_TOKENS="${BASH_REMATCH[1]}"
    SPLIT_VQ_CODEBOOKS="${BASH_REMATCH[2]}"
    CODEBOOK_BINS="${BASH_REMATCH[3]}"
    CODEBOOK_DIM="${BASH_REMATCH[4]}"
    QUANTIZER_ARGS=(
        --quantizer_type split_vq
        --codebook_dim "${CODEBOOK_DIM}"
        --rq_num_tokens "${RQ_TOKENS}"
        --split_vq_num_codebooks "${SPLIT_VQ_CODEBOOKS}"
        --split_vq_codebook_bins "${CODEBOOK_BINS}"
    )
    EVAL_SWEEP_ARGS=()
elif [[ "${CKPT_NAME}" =~ _rq_N([0-9]+)_M([0-9]+)_K([0-9]+)_D([0-9]+) ]]; then
    CODEBOOK_TYPE="rq"
    RQ_TOKENS="${BASH_REMATCH[1]}"
    RQ_STAGES="${BASH_REMATCH[2]}"
    CODEBOOK_BINS="${BASH_REMATCH[3]}"
    CODEBOOK_DIM="${BASH_REMATCH[4]}"
    QUANTIZER_ARGS=(
        --quantizer_type rq
        --codebook_dim "${CODEBOOK_DIM}"
        --rq_num_tokens "${RQ_TOKENS}"
        --rq_num_stages "${RQ_STAGES}"
        --rq_codebook_bins "${CODEBOOK_BINS}"
    )
    EVAL_SWEEP_ARGS=(--eval_rq_active_stages)
    for ((stage = 1; stage <= RQ_STAGES; stage++)); do
        EVAL_SWEEP_ARGS+=("${stage}")
    done
else
    echo "Could not infer quantizer settings from checkpoint directory: ${CKPT_NAME}" >&2
    echo "Expected _split_vq_N<num>_C<num>_K<num>_D<num> or _rq_N<num>_M<num>_K<num>_D<num>." >&2
    exit 2
fi

if [[ ! -f "${CKPT_DIR}/policy_best.ckpt" ]]; then
    echo "Checkpoint not found: ${CKPT_DIR}/policy_best.ckpt" >&2
    exit 2
fi

echo "Checkpoint dir: ${CKPT_DIR}"
echo "Task: ${TASK_NAME}, target camera: ${TARGET_CAMERA}"
echo "Quantizer: ${CODEBOOK_TYPE}"
echo "Quantizer args: ${QUANTIZER_ARGS[*]} ${EVAL_SWEEP_ARGS[*]}"
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
    --lambda_vq 1.0 \
    --lambda_vq_commit 0.25 \
    --vq_warmup_epochs 0 \
    --rq_dead_code_restart_max_fraction 0.05 \
    "${QUANTIZER_ARGS[@]}" \
    "${EVAL_SWEEP_ARGS[@]}" \
    --temporal_agg \
    --eval
