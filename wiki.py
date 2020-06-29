import pandas as pd
import util

wikiir = pd.read_csv('data/ir.concept')
wikiir = wikiir.dropna()
wikiir['concept'] = wikiir['concept'].apply(util.processed)
ir = dict(zip(wikiir['concept'], wikiir['weight']))

wikipsy = pd.read_csv('data/psy.concept')
wikipsy = wikipsy.dropna()
wikipsy['concept'] = wikipsy['concept'].apply(util.processed)
psy = dict(zip(wikipsy['concept'], wikipsy['weight']))

def psy_tfidf(w):
    return psy.get(util.processed(w), 0)

def ir_tfidf(w):
    return ir.get(util.processed(w), 0)
