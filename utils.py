from urllib.request import urlretrieve
from zipfile import ZipFile
import os
from readTrafficSigns import readTrafficSigns
from tqdm import tqdm
import numpy as np
from PIL import Image

def download(url, file):
    """
    Download file from <url>
    :param url: URL to file
    :param file: Local file path
    """
    if not os.path.isfile(file):
        print('Downloading ' + file + '...')
        urlretrieve(url, file)
        print('Download Finished')


def uncompress_features_labels(file, zip_path, data_path):
    print('Extracting ', file, 'to path ', zip_path)
    zf = ZipFile(file, 'r')
    zf.extractall(zip_path)
    zf.close()
    print('Loading image and labels...')
    images, labels = readTrafficSigns(data_path)
    print('Done.')
    return images, labels




