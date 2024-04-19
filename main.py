from sources.config import get_config
from sources.train import train_model

import wandb
import os
from pathlib import Path

def main():
    # Get configuration from option_1
    config = get_config('option_1')
    
    # Configure WandB
    wandb_config = {
        'project': config['EXPERIMENT_NAME'],
        'name': config['RUN_NAME'],
    }
    
    # Login to WandB
    wandb.login()
    
    # Initialize WandB run
    wandb.init(config=config, mode="online", **wandb_config)
    
    # Create save path for models
    SAVE_PATH = os.path.join(wandb.config.RUNS_FOLDER_PTH, wandb.config.RUN_NAME)
    Path(SAVE_PATH).mkdir(parents=True, exist_ok=True)
    
    # Start training the model
    train_model(wandb.config)
    
    # Finish WandB run
    wandb.finish()

if __name__ == '__main__':
    main()
