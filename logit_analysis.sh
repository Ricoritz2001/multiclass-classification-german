#!/bin/bash
#SBATCH --partition=cpu64
#SBATCH --time=12:00:00
#SBATCH --job-name=logit_plotting
#SBATCH --output=Output_file/%j_plotting.out



# Activate the user environment
uenv verbose miniconda3-py39

# Activate your conda environment
conda activate news_classification

# Ensure required packages are installed
pip install pandas matplotlib seaborn

# Set environment variables
export PATH=~/.local/bin:$PATH
TOKENIZERS_PARALLELISM=false

# Run the Python script
python -u -m src.logit_analysis
