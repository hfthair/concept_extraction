import pandas as pd
import math
import gn
# import qc.wiki as wiki

from collections import Counter
import re
import nltk

stop_words = set(nltk.corpus.stopwords.words('english'))
ps = nltk.stem.PorterStemmer()


def split(src):
    tks = [i.strip() for i in re.split(r'[^A-Za-z0-9_]', src) if i.strip()]
    return tks

def processed(text):
    # print(text)
    text = re.sub("[ ]{1,}",r' ',text)
    text = re.sub(r'\W+|\d+', ' ', text.strip().lower())
    tokens = [token.strip()  for token in text.split(" ")]
    tokens = [token for token in tokens if len(token) > 0]
    tokens = [ps.stem(i) for i in tokens if i]

    return " ".join(tokens)

def lower(src):
    src = src.lower()
    tks = split(src)
    return ' '.join(tks)


# book = util.Book()
# other_book = pd.read_csv('../other_books.csv')

# big_book_list = []
# for sec, title, text, keywords in book.tolist():
#     big_book_list.append(title + ' ' + text + ' ' + keywords)
# for text in other_book['text'].tolist():
#     big_book_list.append(text)

# big_book_list = [i for i in big_book_list if len(i) > 300]

books = pd.read_csv('../../student_vector/old_versions/Doc2text.csv')

big_book = '. '.join(books['text'].tolist())
big_book_stem = processed(big_book)

def __book_ngram(N):
    if N <= 0:
        return 0
    tks = split(big_book_stem)
    ngrams = [(i, i+N) for i in range(len(tks)-N)]
    ngrams = [' '.join(tks[i:j]) for i, j in ngrams]
    return len(ngrams)

book_ngram_total = [__book_ngram(i) for i in range(6)]

def tf(tk):
    tk = processed(tk)
    cnt = big_book_stem.count(' ' + tk + ' ')
    return cnt


##############################
# quiz = util.Quiz()
# tmp = []
# for sec, q, ans, wrong, concepts in quiz.tolist():
#     q_ = q + '. \n' + ans
#     tmp.append([q_, concepts])

# dfq = pd.DataFrame(tmp, columns=['q', 'g'])

# max_gram = 6
# def gen_ngrams(q):
#     chunks = util.split_sentences_plus(q)
#     tmp = []
#     for chunk in chunks:
#         tks = util.split(chunk)
#         ngrams = [(i, i+j) for j in range(1, max_gram) for i in range(len(tks))]
#         ngrams = [' '.join(tks[i:j]) for i, j in ngrams]# if tks[i] not in stop_words and tks[i:j][-1] not in stop_words]
#         tmp.extend(ngrams)
#     return util.remove_stem_duplicate(tmp)
# dfq['ngrams'] = dfq['q'].apply(gen_ngrams)

# q_ngrams_all = util.remove_stem_duplicate([j for i in dfq['ngrams'].tolist() for j in i])
# golds = [util.processed(i) for i in util.golds_all()]

# dfr = pd.DataFrame()
# dfr['ngrams'] = q_ngrams_all

# dfr = dfr[~dfr['ngrams'].str.lower().isin(util.stop_words)]

# def calc_truth(w):
#     if processed(w) in golds:
#         return 1.0
#     else:
#         w_ = ' '.join([i for i in w.split(' ') if i not in stop_words])
#         if processed(w_) in golds:
#             return 0.6
#         else:
#             return 0.0

# dfr['TRUE'] = dfr['ngrams'].apply(calc_truth)
# dfr = dfr.set_index('ngrams')


######################################
max_gram = 6
def gen_ngrams(q):
    tks = split(q)
    ngrams = [(i, i+j) for j in range(1, max_gram) for i in range(len(tks))]
    ngrams = [' '.join(tks[i:j]) for i, j in ngrams]# if tks[i] not in stop_words and tks[i:j][-1] not in stop_words]
    # return list(set(ngrams))
    return Counter(ngrams)

GC, GN = (gn.GC, gn.GN)

BC = big_book_stem.count(' ') + 1

def pir_logirge(w):
    return (tf(w)+0.00001) * math.log(((tf(w)+1) / BC) / (gn.cf(w)/GC))

def logir_logirge(w):
    return math.log(tf(w)+0.00001) * math.log(((tf(w)+1) / BC) / (gn.cf(w)/GC))

def logir_logirge2(w, tf2):
    return math.log(tf2+0.00001) * math.log(((tf(w)+1) / BC) / (gn.cf(w)/GC))

def ir_tfidf(w):
    return math.log((1+GN)/(1+gn.df(w))) * tf(w)

def ir_wiki_tfidf(w):
    return wiki.ir_tfidf(w)

def get_top_n_q(q, func=logir_logirge, n=6):
    q = q.lower()
    tks_count = gen_ngrams(q)
    def test(w):
        sl = w.split(' ')
        if sl[0] in stop_words or sl[-1] in stop_words or sl[0].isdigit() or sl[-1].isdigit() or \
            len(sl[0]) == 1 or len(sl[-1]) == 1:
            return False
        else:
            return True
    tks = [i for i in tks_count if i not in stop_words and len(i) > 2 and test(i)]

    pairs = [(func(k), k) for k in tks]
    # pairs = [(func(k), k) for k in tks]
    pairs = sorted(pairs, reverse=True)

    tops = [j for i, j in pairs[:n]]
    return tops


def get_top_n_b(q, func=logir_logirge2, n=6):
    q = q.lower()
    tks_count = gen_ngrams(q)
    def test(w):
        sl = w.split(' ')
        if sl[0] in stop_words or sl[-1] in stop_words or sl[0].isdigit() or sl[-1].isdigit() or \
            len(sl[0]) == 1 or len(sl[-1]) == 1:
            return False
        else:
            return True
    tks = [i for i in tks_count if i not in stop_words and len(i) > 2 and test(i)]

    pairs = [(func(k, tks_count[k]), k) for k in tks]
    # pairs = [(func(k), k) for k in tks]
    pairs = sorted(pairs, reverse=True)

    tops = [j for i, j in pairs[:n]]
    return tops

if __name__ == '__main__':
    dfsec = pd.read_csv('../../student_vector/section2text.csv')
    dfq = pd.read_csv('../../student_vector/quiz_text_section.csv')
    dfsec['extracted'] = dfsec['text'].apply(lambda x: get_top_n_b(x, n=35))
    dfq['extracted'] = dfq['text'].apply(lambda x: get_top_n_q(x, n=5))

    dfsec[['section', 'extracted']].to_csv('d:/sv/section2text_lei.csv', index=False)
    dfq.to_csv('d:/sv/quiz_text_section_lei.csv', index=False)
