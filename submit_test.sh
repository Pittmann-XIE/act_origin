#!/bin/bash
#SBATCH --job-name=test_DINO
#SBATCH --output=slurm_test_%j.out   
#SBATCH --error=slurm_test_%j.err    
#SBATCH --partition=testing          # Force it to the testing queue
#SBATCH --nodes=1                    
#SBATCH --ntasks=1                   
#SBATCH --cpus-per-task=4            # Match your DataLoader workers
#SBATCH --mem=64G                    # 16GB is plenty for a CPU test
#SBATCH --time=00:30:00              # 30 minute limit (Testing queue max is 1hr)

source ~/.bashrc
conda activate python311t

export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa

echo "Starting CPU-only test run on node: $SLURM_JOB_NODELIST"

# Run the script with a tiny batch size and only 1 epoch
python -u imitate_episodes_bisim_A.py \
    --task_name sim_transfer_cube_scripted \
    --ckpt_dir ./checkpoints/checkpoints_bisim_A_1 \
    --policy_class ACT \
    --kl_weight 10 \
    --chunk_size 100 \
    --hidden_dim 512 \
    --batch_size 64 \
    --dim_feedforward 3200 \
    --lr 2e-5 \
    --seed 10 \
    --num_epochs 4
