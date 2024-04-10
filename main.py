from sources.config import configuration
from sources.train import train_model

import wandb
from pathlib import Path

def main():
    wandb.login()
    wandb.init(config = configuration,
            project = "transformer-from-scratch",
            mode = "disabled")
    
    Path(wandb.config.RUNS_FOLDER_PTH).mkdir(parents = True, exist_ok = True)
    Path(wandb.config.SAVE_MODEL_DIR).mkdir(parents = True, exist_ok = True)
    train_model(wandb.config)