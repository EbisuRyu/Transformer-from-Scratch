import os

option_1 = dict(
    # RUN CONFIG:
    EXPERIMENT_NAME = 'transformer-from-scratch',
    RUN_NAME = 'official_single_gpu_run',
    RUN_DESCRIPTION = 'Default run on GPU, 10GB of VRAM needed for this.',
    RUNS_FOLDER_PTH = './Transformer-from-Scratch/model/runs',
    
    # DATA CONFIG:
    DATASET_SIZE = 133317,
    TEST_PROPORTION = 0.001,
    MAX_LEN = 40,
    VOCAB_SIZE = 60000,
    TOKENIZER_TYPE = 'wordlevel', # 'wordlevel' or 'bpe
    
    # TRAINING CONFIG:
    BATCH_SIZE = 48,
    GRAD_ACCUMULATION_STEPS = 2048//48,
    WORKER_COUNT = 10,
    EPOCHS = 100,
    
    # OPTIMIZER CONFIG:
    BETAS = (0.9, 0.98),
    EPS = 1e-9,
    
    # SCHEDULER CONFIG:
    N_WARMUP_STEPS = 4000,
    
    # MODEL CONFIG:
    D_MODEL = 512,
    NUM_LAYERS = 6,
    N_HEADS = 8,
    D_FF = 2048,
    DROPOUT = 0.1,
    
    # LOCAL PATH CONFIG:
    PRETRAIN_MODEL_PTH = './Transformer-from-Scratch/model/model_ckpt_epoch150.pt',
    PRETRAIN_TOKENIZER_PT = './Transformer-from-Scratch/model/runs/official_single_gpu_run/tokenizer.json',
    SAVE_MODEL_DIR = './model',
    TOKENIZER_SAVE_PTH = './Transformer-from-Scratch/model/runs/official_single_gpu_run/tokenizer.json',
    
    # OTHER:
    DEVICE = 'gpu',
    MODEL_SAVE_EPOCH_CNT = 5,
    LABEL_SMOOTHING = 0.1   
)

option_2 = dict(
    # RUN CONFIG:
    EXPERIMENT_NAME = 'transformer-from-scratch',
    RUN_NAME = 'unofficial_single_gpu_run',
    RUN_DESCRIPTION = 'Default run on GPU, 10GB of VRAM needed for this.',
    RUNS_FOLDER_PTH = './Transformer-from-Scratch/runs',
    
    # DATA CONFIG:
    DATASET_SIZE = 30,
    TEST_PROPORTION = 0.001,
    MAX_LEN = 40,
    VOCAB_SIZE = 60000,
    TOKENIZER_TYPE = 'wordlevel', # 'wordlevel' or 'bpe
    
    # TRAINING CONFIG:
    BATCH_SIZE = 48,
    GRAD_ACCUMULATION_STEPS = 2048//48,
    WORKER_COUNT = 10,
    EPOCHS = 50,
    
    # OPTIMIZER CONFIG:
    BETAS = (0.9, 0.98),
    EPS = 1e-9,
    
    # SCHEDULER CONFIG:
    N_WARMUP_STEPS = 4000,
    
    # MODEL CONFIG:
    D_MODEL = 512,
    NUM_LAYERS = 6,
    N_HEADS = 8,
    D_FF = 2048,
    DROPOUT = 0.1,
    
    # LOCAL PATH CONFIG:
    PRETRAIN_MODEL_PTH = './Transformer-from-Scratch/model/model_ckpt_epoch150.pt',
    PRETRAIN_TOKENIZER_PT = './Transformer-from-Scratch/model/tokenizer.json',
    
    # OTHER:
    DEVICE = 'cpu',
    MODEL_SAVE_EPOCH_CNT = 10,
    LABEL_SMOOTHING = 0.1   
)

dict_of_options = {
    'option_1': option_1,
    'option_2': option_2
}

def get_config(option):
    return dict_of_options[option]