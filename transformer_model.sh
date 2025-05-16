#!/bin/bash
#SBATCH --gres=gpu:0
#SBATCH --partition=gpu
#SBATCH --time=48:00:00
#SBATCH --job-name=checkthat_training
#SBATCH --output=/home/stud/ricardor/bhome/bachelor/German-News-Data-Classification/Output_file/%j.out

# Load CUDA and cuDNN modules
module load cuda/12.2.0 cudnn/8.8.0

# Activate the user environment (uenv)
uenv verbose cuda-12.2.0 cudnn-12.x-8.8.0
uenv miniconda3-py39

# Activate the Conda environment
conda activate news_classification

# Install tf-keras to ensure compatibility with Transformers
pip install tf-keras

# (Optional) Install TensorFlow if needed
pip install tensorflow

# Add user's local bin directory to the PATH
export PATH=~/.local/bin:$PATH
echo "PATH: $PATH"
TOKENIZERS_PARALLELISM=false 

# Run the transformer training script
python -u -m src.train_transformer_models
