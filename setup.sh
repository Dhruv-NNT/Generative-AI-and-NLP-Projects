#!/bin/bash
#SBATCH --partition=SCSEGPU_M1
#SBATCH --qos=q_amsai
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=12G
#SBATCH --job-name=dlassigns
#SBATCH --output=output_%x_%j.out
#SBATCH --error=error_%x_%j.err

export CUBLAS_WORKSPACE_CONFIG=:16:8

module load anaconda3/23.5.2
eval "$(conda shell.bash hook)"
conda activate venv_nlp


pip install --upgrade transformers
pip install --upgrade datasets
pip install --upgrade accelerate
pip install --upgrade bitsandbytes
pip install --upgrade peft
pip install numpy
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu1180
pip install -q -U datasets scipy ipywidgets
pip install --upgrade wandb
pip install scikit-learn
pip install tqdm
pip install trl
pip install tensorboard
pip install streamlit

