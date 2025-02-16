#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --time=36:00:00
#SBATCH --mem=32gb
#SBATCH --partition=gpu_cuda
#SBATCH --gres=gpu:h100:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zhiyong.ma@student.uq.edu.au
#SBATCH --no-requeue

#SBATCH --job-name=eval_noise_llama3_ppl
#SBATCH -o eval_llama3_noise_ppl_output.log
#SBATCH -e eval_llama3_noise_ppl_error.log

source /home/uqzwan39/.bashrc
conda activate opencompass_lmdeploy
export HF_CACHE=/scratch/user/uqzwan39/zhiyong.ma/models/
export HF_DATASETS_CACHE=/scratch/user/uqzwan39/zhiyong.ma/datasets/
srun --export=PATH,TERM,HOME,LANG python -u run.py configs/eval_noise_llama3.py
