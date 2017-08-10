from urllib.request import urlretrieve
from zipfile import ZipFile
import os

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


def uncompress(file, zip_path):
    print('Extracting ', file, 'to path ', zip_path)
    zf = ZipFile(file, 'r')
    zf.extractall(zip_path)
    zf.close()

def PrepareData():
    fn = 'https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip'
    target_fn = 'traffic-sign-data.zip'
    if not os.path.isfile(target_fn):
        download(fn, target_fn)
        uncompress(target_fn, './')
    print('All images and labels uncompressed.')


