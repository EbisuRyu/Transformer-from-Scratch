from sources.config import get_config
from sources.dataset import get_translation_dataloaders
from sources.machine_translation import MachineTranslationTransformer
from sources.scheduler import CustomScheduler
from sources.learner import Learner
from sources.train import train_model

import torch
import torch.nn as nn
import torch.optim as optim
import os
from tokenizers import Tokenizer
from pathlib import Path

config = get_config()
Path(config.RUNS_FOLDER_PTH).mkdir(parents = True, exist_ok = True)
Path(config.SAVE_MODEL_DIR).mkdir(parents = True, exist_ok = True)
train_model(config)