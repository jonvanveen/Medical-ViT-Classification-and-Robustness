#!/usr/bin/env zsh
#SBATCH --job-name=swin_chxray_base_model
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
pip install pandas
pip install git+https://github.com/fra31/auto-attack
pip install scikit-image
pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

# Pretrained swin
#python -m torch.distributed.launch --nproc_per_node 8 --master_port 12345 main.py --cfg configs/swin/swin_tiny_patch4_window7_224.yaml --resume swin_tiny_patch4_window7_224.pth --data-path ~/Data/chxray/imagenet --batch-size 128  --opts TRAIN.EPOCHS 200 TRAIN.WARMUP_EPOCHS 10 TRAIN.BASE_LR 0.0002 --output ~/Models/swin_chxray_base_output

# data_dir is for autoattack, data_path for swin
python -m torch.distributed.launch --nproc_per_node 8 --master_port 12345 ~/Models/Swin-Transformer/swin_autoattack.py --cfg=configs/swin/swin_tiny_patch4_window7_224.yaml --data_dir=~/Data/chxray/auto_1000 --data_path=~/Data/chxray/imagenet --csv=~/Data/chxray/autoattack_1000.csv --model=ckpt_epoch_150.pth --save_dir=~/Models/swin_chxray_adv_output/results --batch_size=8 --num_workers=2 --log_path=~/Models/swin_chxray_adv_outputs/results/logger.txt #--zip=False --cache_mode='no' --pretrained=False --resume=False --accumulation_steps=0 --use_checkpoint=False --amp_opt_level='O1' --disable_amp=False --output=~/Models/swin_chxray_adv_outputs/results --tag=False --eval=False --throughput=False --fused_window_process=False --local_rank=0 #--log_path ~/Models/swin_chxray_base_output/swin_tiny_window7_224/default/log_rank7.txt
