import gdown
def download_pretrained_model(epoch = 60):
    url = 'https://drive.google.com/uc?id=1Be1791q3DZj8YRwXHJgRZ14v48B5KRat'
    output = f'/content/Transformer-from-Scratch/model/model_ckpt_epoch{epoch}.pt'
    gdown.download(url, output, quiet = False)
download_pretrained_model()