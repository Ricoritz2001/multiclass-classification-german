#!/bin/bash
#SBATCH --partition=cpu64
#SBATCH --time=96:00:00
#SBATCH --job-name=classic_training
#SBATCH --output=Output_file/%j.out


# Activate the user environment (if using uenv, adjust as necessary)
uenv verbose miniconda3-py39

# Activate the Conda environment
conda activate news_classification

# Install necessary packages (spaCy and NumPy) if not already installed
pip install spacy numpy

# Download the German spaCy model (de_core_news_md) if not already installed
python -m spacy download de_core_news_lg

# Add user's local bin directory to the PATH
export PATH=~/.local/bin:$PATH
echo "PATH: $PATH"
TOKENIZERS_PARALLELISM=false

# Run the classical models training script using the module flag
python -u -m src.train_classical_model
