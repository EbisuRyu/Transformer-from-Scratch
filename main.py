from sources.config import get_config
from sources.train import train_model

import wandb
from pathlib import Path

def main():
        
    config = get_config('option_1')
    wandb_config = {
            'project': config['EXPERIMENT_NAME'],
            'name': config['RUN_NAME'],
            'description': config['RUN_DESCRIPTION']
    }
    wandb.login()
    wandb.init(config = config, mode = "online", **wandb_config)
    
    Path(wandb.config.RUNS_FOLDER_PTH).mkdir(parents = True, exist_ok = True)
    Path(wandb.config.SAVE_MODEL_DIR).mkdir(parents = True, exist_ok = True)
    train_model(wandb.config)
    wandb.finish()

if __name__ == '__main__':
        main()