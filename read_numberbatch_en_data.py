###unzip the file
import gzip
import shutil
import os

path="../data"
os.chdir(path)

import gzip
import shutil
f_in=open
with gzip.open('numberbatch-en-19.08.txt.gz', 'rb') as f_in:
    with open('numberbatch-en.txt','wb') as f_out:
        shutil.copyfileobj(f_in,f_out)