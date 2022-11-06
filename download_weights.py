import gdown
import zipfile
import os

def exractfiles(file, dest):
    with zipfile.ZipFile(file, 'r') as zip_ref:
        zip_ref.extractall(dest)


def download_model():
    model_url = '1rbQtfTpDIiyRxsrW5uQbfSxnvzcN76Ab'
    yolor = '1KEanvTOJoMlMDw0A-6Nw7nXmVylhnJu-'
    url = f'https://drive.google.com/uc?id={model_url}'
    gdown.download(url,output='weights.zip', quiet=False)  

    url = f'https://drive.google.com/uc?id={yolor}'
    gdown.download(url,output='yolor_p6.pt', quiet=False)  
    
    deepsort = '1VZ05gzg249Q1m8BJVQxl3iHoNIbjzJf8'
    url = f'https://drive.google.com/uc?id={deepsort}'
    gdown.download(url,output='ckpt.zip', quiet=False)  
    

    print('Download Completed Successfully!')

    print('Extracting and Moving Weights....')
    exractfiles('weights.zip', './yolov7/')
    exractfiles('ckpt.zip', './deep_sort_pytorch/deep_sort/deep/checkpoint/')
    print('Extracting and Moving Weights Completed!')
    os.remove('weights.zip')
    os.remove('ckpt.zip')

if __name__ == "__main__":
    download_model()