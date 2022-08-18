#!/usr/bin/env zsh
#SBATCH --job-name=resnet_base
#SBATCH --partition=research
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=0-48:00:00
#SBATCH --output="benchmark_1_core_1_gpu-%j.txt"
#SBATCH -G 8

cd $SLURM_SUBMIT_DIR

module load anaconda/full
bootstrap_conda
conda activate swin
pip install timm==0.4.12 yacs==0.1.8
pip install -U PyYAML
pip install matplotlib sklearn

python resnet50_base.py
