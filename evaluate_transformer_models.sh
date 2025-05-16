#!/bin/bash
#SBATCH --gres=gpu:0
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --job-name=checkthat_evaluation
#SBATCH --output=Output_file/%j.out

# Load CUDA and cuDNN modules
module load cuda/12.2.0 cudnn/8.8.0

# Activate the user environment (uenv)
uenv verbose cuda-12.2.0 cudnn-12.x-8.8.0
uenv miniconda3-py39

# Activate the Conda environment
conda activate news_classification

# Install matplotlib if it's not already installed
pip install matplotlib

pip install seaborn

# Add user's local bin directory to the PATH
export PATH=~/.local/bin:$PATH
echo "PATH: $PATH"

# Disable tokenizers parallelism warnings
export TOKENIZERS_PARALLELISM=false 

# Run the evaluation script as a module
python -u -m src.evaluate_transformer_models

