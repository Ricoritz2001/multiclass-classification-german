#!/bin/bash
#SBATCH --gres=gpu:0
#SBATCH --partition=gpu
#SBATCH --time=1:00:00
#SBATCH --job-name=news_classification_setup
#SBATCH --output=news_classification_setup.out

# Load CUDA and cuDNN modules
module load cuda/12.2.0 cudnn/8.8.0

# Activate the user environment (uenv)
uenv verbose cuda-12.2.0 cudnn-12.x-8.8.0
uenv miniconda3-py39

# Create and activate the Conda environment for news_classification
conda create -n news_classification -c pytorch pytorch torchvision torchaudio pytorch-cuda=12.1 -c nvidia -y
conda activate news_classification

# Install Python packages
pip install torch torchvision torchaudio
pip install transformers[torch]
pip install -r requirements.txt
