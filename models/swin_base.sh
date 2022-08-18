#!/usr/bin/env zsh
#SBATCH --job-name=swin_chxray_base_model
#SBATCH --partition=wacc
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=0-24:00:00
#SBATCH --output="benchmark_1_core_1_gpu-%j.txt"
#SBATCH -G 4

cd $SLURM_SUBMIT_DIR

module load anaconda/full
bootstrap_conda
conda activate swin

#conda update -n base -c defaults conda
#conda install pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=10.2 -c pytorch
#pip install timm==0.4.12 opencv-python==4.4.0.46 termcolor==1.1.0 yacs==0.1.8

#cd kernels/window_process
#python setup.py install #--user

# Pretrained swin
python -m torch.distributed.launch --nproc_per_node 4 --master_port 12345 main.py --cfg configs/swin/swin_tiny_patch4_window7_224.yaml --resume ~/Models/swin_chxray_base_output/swin_tiny_patch4_window7_224/default/ckpt_epoch_40.pth --data-path ~/Data/chxray/imagenet --batch-size 128  --opts TRAIN.EPOCHS 200 TRAIN.WARMUP_EPOCHS 10 TRAIN.BASE_LR 0.0002 --output ~/Models/swin_chxray_base_output

# NOT Swin train from scratch
#python -m torch.distributed.launch --nproc_per_node 8 --master_port 12345 main.py --eval --cfg configs/swin/swin_tiny_patch4_window7_224.yaml --resume swin_tiny_patch4_window7_224.pth --data-path ~/Data/chxray/imagenet --batch-size 128  --opts TRAIN.EPOCHS 200 TRAIN.WARMUP_EPOCHS 10 TRAIN.BASE_LR 0.0002 --output ~/Models/swin_chxray_base_output

python test.py
