import numpy
import math
import sqlite3
import re
import pandas as pd


DB_FILE_NAME = 'articles.db'

# LSH signature generation using random projection
def get_signature(user_vector, rand_proj): 
    res = 0
    for p in (rand_proj):
        res = res << 1
        val = numpy.dot(p, user_vector)
        if val >= 0:
            res |= 1
    return res

# get number of '1's in binary
# running time: O(# of '1's)
def nnz(num):
    if num == 0:
        return 0
    res = 1
    num = num & (num-1)
    while num:
        res += 1
        num = num & (num-1)
    return res     

# angular similarity using definitions
# http://en.wikipedia.org/wiki/Cosine_similarity
def angular_similarity(a,b):
    dot_prod = numpy.dot(a,b)
    sum_a = sum(a**2) **.5
    sum_b = sum(b**2) **.5
    cosine = dot_prod/sum_a/sum_b # cosine similarity
    theta = math.acos(cosine)
    return 1.0-(theta/math.pi)

def build_df() -> pd.DataFrame:
    """Build dataframe with derived fields."""
    with closing(sqlite3.connect(DB_FILE_NAME)) as conn:
        articles = pd.read_sql_query('select * from articles', conn)

    articles = articles.replace([None], [''], regex=True)
    articles['word_count'] = articles.apply(count_words, axis=1)
    return articles


if __name__ == '__main__':
    dim = 200 # number of dimensions per data
    d = 2**10 # number of bits per signature
    
    nruns = 24 # repeat times
    
    avg = 0
    articles = build_df()

    user1 = numpy.random.randn(dim)
    user2 = numpy.random.randn(dim)
    randv = numpy.random.randn(d, dim)
    r1 = get_signature(user1, randv)
    r2 = get_signature(user2, randv)
    xor = r1^r2
    true_sim, hash_sim = (angular_similarity(user1, user2), (d-nnz(xor))/float(d))
    diff = abs(hash_sim-true_sim)/true_sim
    avg += diff
    print ('true %.4f, hash %.4f, diff %.4f' % (true_sim, hash_sim, diff) )
