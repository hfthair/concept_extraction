import util
import pandas as pd


df = pd.read_csv('data/cmubook.csv', encoding='utf8')

big_book_list = (df['title'] + '\n' + df['text']).tolist()

big_book = '. '.join(big_book_list)
big_book_stem = util.processed(big_book)

def __book_ngram(N):
    if N <= 0:
        return 0
    tks = util.split(big_book_stem)
    ngrams = [(i, i+N) for i in range(len(tks)-N)]
    ngrams = [' '.join(tks[i:j]) for i, j in ngrams]
    return len(ngrams)

book_ngram_total = [__book_ngram(i) for i in range(6)]

def tf(tk):
    tk = util.processed(tk)
    cnt = big_book_stem.count(' ' + tk + ' ')
    return cnt


##############################

dfq = pd.read_csv('data/quiz.small.csv', encoding='utf8')
dfq['g'] = 0

max_gram = 6
def gen_ngrams(q):
    chunks = util.split_sentences_plus(q)
    tmp = []
    for chunk in chunks:
        tks = util.split(chunk)
        ngrams = [(i, i+j) for j in range(1, max_gram) for i in range(len(tks))]
        ngrams = [' '.join(tks[i:j]) for i, j in ngrams]# if tks[i] not in stop_words and tks[i:j][-1] not in stop_words]
        tmp.extend(ngrams)
    return util.remove_stem_duplicate(tmp)
dfq['ngrams'] = dfq['q'].apply(gen_ngrams)

q_ngrams_all = util.remove_stem_duplicate([j for i in dfq['ngrams'].tolist() for j in i])
golds = []

dfr = pd.DataFrame()
dfr['ngrams'] = q_ngrams_all

dfr = dfr[~dfr['ngrams'].str.lower().isin(util.stop_words)]

def calc_truth(w):
    if util.processed(w) in golds:
        return 1.0
    else:
        w_ = ' '.join([i for i in w.split(' ') if i not in util.stop_words])
        if util.processed(w_) in golds:
            return 0.6
        else:
            return 0.0

dfr['TRUE'] = dfr['ngrams'].apply(calc_truth)
dfr = dfr.set_index('ngrams')


