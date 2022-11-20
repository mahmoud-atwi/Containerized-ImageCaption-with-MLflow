# data available on the below link
# https://drive.google.com/file/d/1izC_f_HqfqdXwJA8HNderfr6Z2PDn8j4/view?usp=sharing

import gdown
from zipfile import ZipFile

url = 'https://drive.google.com/uc?id=1J8UB8ho7bfUg7btAt6COjCFVYHRgw-N-'
data_file = 'data.zip'
gdown.download(url, data_file, quiet=False)

def extract(file, dir):
    with ZipFile(file, 'r') as zObject:
        zObject.extractall(dir)

if __name__ == "main":
    extract(data_file, dir='/data/')