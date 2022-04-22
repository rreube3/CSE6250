###unzip the file
import os
import gzip
import shutil
f_in=open
with gzip.open('data/numberbatch-en-19.08.txt.gz', 'rb') as f_in:
    with open('data/numberbatch-en.txt','wb') as f_out:
        shutil.copyfileobj(f_in,f_out)
