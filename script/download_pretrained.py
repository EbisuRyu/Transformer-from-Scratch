import gdown
def download_pretrained_model(epoch = 20):
    url = 'https://drive.google.com/uc?id=1cbhg7wphS8OnM08Fg1SFpp8d_Z11W8cV'
    output = f'/kaggle/working/Transformer-from-Scratch/model/model_ckpt_epoch{epoch}.pth'
    gdown.download(url, output, quiet = False)
    
download_pretrained_model()