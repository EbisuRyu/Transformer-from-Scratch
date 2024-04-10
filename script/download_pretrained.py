import gdown
def download_pretrained_model(epoch = 60):
    url = 'https://drive.google.com/uc?id=1U5VVB8n0H93Y0lVwgqnYP3QGZMehu6sH'
    output = f'/content/Transformer-from-Scratch/model/model_ckpt_epoch{epoch}.pt'
    gdown.download(url, output, quiet = False)
download_pretrained_model()