configuration = dict(
    EXPERIMENT_NAME = 'runs/tmodel',
    RUN_NAME = 'unofficial_single_gpu_run',
    RUN_DESCRIPTION = 'Default run on GPU, 10GB of VRAM needed for this.',
    RUNS_FOLDER_PTH = './Transformer-from-Scratch/runs',
    SAVE_MODEL_DIR = './Transformer-from-Scratch/model',
    PRETRAIN_MODEL_PTH = './Transformer-from-Scratch/model/model_ckpt_epoch150.pt',
    PRETRAIN_TOKENIZER_PT = './Transformer-from-Scratch/model/tokenizer.json',
    TOKENIZER_SAVE_PTH = './Transformer-from-Scratch/model/tokenizer.json',
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
    # OTHER:
    MODEL_SAVE_EPOCH_CNT = 10,
    DEVICE = 'cpu',
    LABEL_SMOOTHING = 0.1   
)