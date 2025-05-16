#!/bin/bash
#SBATCH --partition=cpu64
#SBATCH --time=24:00:00
#SBATCH --job-name=classic_evaluation
#SBATCH --output=Output_file/%j.out

# Activate the user environment
uenv verbose miniconda3-py39

# Activate the Conda environment
conda activate news_classification

# Install required packages if not already installed
pip install matplotlib seaborn scikit-learn

# Add user's local bin directory to the PATH
export PATH=~/.local/bin:$PATH
export TOKENIZERS_PARALLELISM=false

# Run the classical model evaluation script
python -u -m src.evaluate_classical_models
