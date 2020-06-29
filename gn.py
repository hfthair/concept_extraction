import pandas as pd

import re
import nltk

stop_words = set(nltk.corpus.stopwords.words('english'))
ps = nltk.stem.PorterStemmer()

def split(src):
    tks = [i.strip() for i in re.split(r'[^A-Za-z0-9_]', src) if i.strip()]
    return tks

def processed(src):
    src = src.lower()
    tks = split(src)
    tks = [ps.stem(i) for i in tks if i]
    return ' '.join(tks)


GC = 468491999592
GN = 4541627
# tmp = pd.read_csv('data/one.csv', delimiter='\t', header=None)
tmp = pd.read_csv('data/one.csv', delimiter='\t', header=None)

tmp.columns = ['w', 'cf', 'df']
tmp = tmp.groupby("w").max()
gdf = dict(zip(tmp.index, tmp.df))
gcf = dict(zip(tmp.index, tmp.cf))
def df(tk):
    tk = processed(tk)
    return gdf.get(tk, 20)

def cf(tk):
    tk = processed(tk)
    return gcf.get(tk, 20)

def cfx(tk):
    tk = processed(tk)
    r = gcf.get(tk, 0)
    if r < 1:
        p = 1 * GC
        for t in tk.split(' '):
            freq = gcf.get(t, 1)
            p = p * freq / GC
        r = p
    return r

def dfx(tk):
    tk = processed(tk)
    r = gdf.get(tk, 0)
    if r < 1:
        p = 1 * GN
        for t in tk.split(' '):
            freq = gdf.get(t, 1)
            p = p * freq / GN
        r = p
    return r
