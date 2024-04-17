import torch
import wandb
import logging
import torch.nn as nn
import torch.optim as optim
from tokenizers import Tokenizer

from .learner import Learner
from .scheduler import CustomScheduler
from .dataset import get_translation_dataloaders
from .machine_translation import MachineTranslationTransformer

# Configure log
log = logging.getLogger(__name__)
logging.basicConfig(level = logging.INFO) 

def train_model(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
   
    train_dataloader, val_dataloader = get_translation_dataloaders(
            dataset_size = config.DATASET_SIZE,
            vocab_size = config.VOCAB_SIZE,
            tokenizer_save_pth = config.TOKENIZER_SAVE_PTH,
            tokenizer_type = config.TOKENIZER_TYPE,
            batch_size = config.BATCH_SIZE,
            max_len = config.MAX_LEN,
            test_proportion = config.TEST_PROPORTION,
        )

    model = MachineTranslationTransformer(
            d_model = config.D_MODEL,
            num_layers = config.NUM_LAYERS,
            src_vocab_size = config.VOCAB_SIZE,
            trg_vocab_size = config.VOCAB_SIZE,
            n_heads = config.N_HEADS,
            d_ff = config.D_FF,
            dropout = config.DROPOUT
        ).to(device)
    
    wandb.watch(model, log = "all", log_freq = 1000)
    loss_func = nn.CrossEntropyLoss(ignore_index = 0, label_smoothing = 0.1, reduction = 'mean')
    optimizer = optim.Adam(model.parameters(), betas = config.BETAS, eps = config.EPS)
    initial_epoch = 0
    loss_func = nn.CrossEntropyLoss(ignore_index = 0, label_smoothing = 0.1, reduction = 'mean')
    optimizer = optim.Adam(model.parameters(), betas = config.BETAS, eps = config.EPS)
    if config.PRETRAIN_MODEL_PTH is not None:
        model_filename = config.PRETRAIN_MODEL_PTH
        print(f'Loading model weights from {model_filename}')
        state = torch.load(model_filename)
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        model.load_state_dict(state['model_state_dict'])
    
    scheduler = CustomScheduler(optimizer, config.D_MODEL, config.N_WARMUP_STEPS)
    
    tokenizer = Tokenizer.from_file(config.TOKENIZER_SAVE_PTH)
    learner = Learner(config, model, tokenizer, train_dataloader, val_dataloader, loss_func, optimizer, scheduler, device)
    learner.fit(initial_epoch, config.EPOCHS)