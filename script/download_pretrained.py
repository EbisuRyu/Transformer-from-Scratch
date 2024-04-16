import gdown
from pathlib import Path

def download_pretrained_model(epoch = 150):
    url = 'https://drive.google.com/uc?id=1Be1791q3DZj8YRwXHJgRZ14v48B5KRat'
    output = f'./Transformer-from-Scratch/model/model_ckpt_epoch{epoch}.pt'
    Path('./Transformer-from-Scratch/model/').mkdir(parents = True, exist_ok = True)
    gdown.download(url, output, quiet = False)
download_pretrained_model()