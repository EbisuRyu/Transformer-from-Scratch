from pathlib import Path

class Configuration():
    def __init__(self):
        self.EXPERIMENT_NAME = 'runs/tmodel'
        self.RUN_NAME = 'unofficial_overfit_cpu_run'
        self.RUN_DESCRIPTION = """
            This run is for testing can the model overfit a single example.
            It is useful when debugging.
            For better results change the scheduler in train.py.
            """
        self.RUN_NAME = 'unofficial_single_gpu_run'
        self.RUN_DESCRIPTION = 'Default run on GPU, 10GB of VRAM needed for this.'
        self.RUNS_FOLDER_PTH = './runs'
        self.SAVE_MODEL_DIR = './model'
        self.PRETRAIN_MODEL_PTH = None #'../model/pretrain.pth'
        self.TOKENIZER_SAVE_PTH = './model/tokenizer.json'
        # DATA CONFIG:
        self.DATASET_SIZE = 100000
        self.TEST_PROPORTION = 0.001
        self.MAX_LEN = 40
        self.VOCAB_SIZE = 60000
        self.TOKENIZER_TYPE = 'wordlevel' # 'wordlevel' or 'bpe
        # TRAINING CONFIG:
        self.BATCH_SIZE = 48
        self.GRAD_ACCUMULATION_STEPS = 2048//48
        self.WORKER_COUNT = 10
        self.EPOCHS = 100
        # OPTIMIZER CONFIG:
        self.BETAS = (0.9, 0.98)
        self.EPS = 1e-9
        # SCHEDULER CONFIG:
        self.N_WARMUP_STEPS = 4000
        # MODEL CONFIG:
        self.D_MODEL = 512
        self.NUM_LAYERS = 6
        self.N_HEADS = 8
        self.D_FF = 2048
        self.DROPOUT = 0.1
        # OTHER:
        self.MODEL_SAVE_EPOCH_CNT = 10
        self.DEVICE = 'cpu'
        self.LABEL_SMOOTHING = 0.1

def get_config():
    return Configuration()