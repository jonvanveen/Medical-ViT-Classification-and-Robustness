# Shell script to execute baseline Swin-T code on the WACC Euler compute cluster
# Specifies memory, runtime, and GPUs for job

#!/usr/bin/env zsh
#SBATCH --job-name=swin_chxray_base_model
#SBATCH --partition=wacc
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=0-24:00:00
#SBATCH --output="benchmark_1_core_1_gpu-%j.txt"
#SBATCH -G 8

cd $SLURM_SUBMIT_DIR

module load anaconda/full
bootstrap_conda
conda activate swin

# Pretrained swin
python -m torch.distributed.launch --nproc_per_node 8 --master_port 12345 main.py --cfg configs/swin/swin_tiny_patch4_window7_224.yaml --resume ~/Models/swin_chxray_base_output/swin_tiny_patch4_window7_224/default/ckpt_epoch_40.pth --data-path ~/Data/chxray/imagenet --batch-size 128  --opts TRAIN.EPOCHS 200 TRAIN.WARMUP_EPOCHS 10 TRAIN.BASE_LR 0.0002 --output ~/Models/swin_chxray_base_output
