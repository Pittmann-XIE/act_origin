#!/bin/bash
#SBATCH --job-name=Bi100
#SBATCH --output=slurm_train_%j.out   # Standard output log (%j inserts job ID)
#SBATCH --error=slurm_train_%j.err    # Standard error log
#SBATCH --nodes=1                     # Run everything on one machine
#SBATCH --ntasks=1                    # One main training task
#SBATCH --cpus-per-task=4             # 4 workers * 1 GPUs = 4 CPUs
#SBATCH --gres=gpu:1                 # Request ALL 1 GPUs
#SBATCH --mem=64G                     # Request 64GB of System RAM
#SBATCH --time=72:00:00               # Max run time (hours)

# Print some basic info
echo "Job running on node: $SLURM_JOB_NODELIST"
echo "Starting training..."

# IMPORTANT: Load your environment here if needed (e.g., conda)
source ~/.bashrc
conda activate python311t
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa

# Run the script with your desired arguments
python -u imitate_episodes_bisim_B.py\
    --task_name sim_transfer_cube_scripted \
    --ckpt_dir ./checkpoints/checkpoints_sim_transfer_cube_bisim_B_100 \
    --policy_class ACT \
    --kl_weight 10 \
    --chunk_size 100 \
    --hidden_dim 512 \
    --batch_size 16 \
    --dim_feedforward 3200 \
    --lr 2e-5 \
    --seed 10 \
    --num_epochs 4000

