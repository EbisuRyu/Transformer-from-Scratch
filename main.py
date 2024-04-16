from sources.config import get_config
from sources.train import train_model

import wandb
import os
from pathlib import Path

def main():
        
    config = get_config('option_1')
    wandb_config = {
            'project': config['EXPERIMENT_NAME'],
            'name': config['RUN_NAME'],
    }
    wandb.login()
    wandb.init(config = config, mode = "online", **wandb_config)
    SAVE_PATH = os.path.join(wandb.config.RUNS_FOLDER_PTH, wandb.config.RUN_NAME)
    Path(SAVE_PATH).mkdir(parents = True, exist_ok = True)
    train_model(wandb.config)
    wandb.finish()

if __name__ == '__main__':
        main()