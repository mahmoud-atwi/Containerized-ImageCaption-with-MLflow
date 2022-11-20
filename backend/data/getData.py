import gdown

url = 'https://drive.google.com/uc?id=1J8UB8ho7bfUg7btAt6COjCFVYHRgw-N-'
output = 'data.zip'
gdown.download(url, output, quiet=False)

! unzip data.zip > /data/