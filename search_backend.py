import sys
from collections import Counter, OrderedDict,defaultdict
import itertools
from itertools import islice, count, groupby
import pandas as pd
import os
import re
from operator import itemgetter
from time import time
from timeit import timeit
from pathlib import Path
import pickle
import pandas as pd
import numpy as np
from google.cloud import storage
import math
import builtins
from inverted_index_gcp import MultiFileReader
# from inverted_index_colab import MultiFileReader
from contextlib import closing
from nltk import LancasterStemmer
import math

# When preprocessing the data have a dictionary of document length for each document saved in a variable called `DL`.
class BM25_from_index:
    """
    Best Match 25.
    ----------
    k1 : float, default 1.5

    b : float, default 0.75

    index: inverted index
    """

    def __init__(self, index, k1=1.5, b=0.75, is_bi=False, stem=False):
        self.b = b
        self.k1 = k1
        self.index = index
        self.N = len(index.NF)
        self.AVGDL = builtins.sum([v[1] for v in index.NF.values()]) / self.N
        self.is_bi = is_bi
        self.stem = stem
        self.lancaster = LancasterStemmer()

    def get_top_n(self, score_dict, N=3):
        """
        Sort and return the highest N documents according to the cosine similarity score.
        Generate a dictionary of cosine similarity scores

        Parameters:
        -----------
        sim_dict: a dictionary of similarity score as follows:
                                                                    key: document id (e.g., doc_id)
                                                                    value: similarity score. We keep up to 5 digits after the decimal point. (e.g., round(score,5))

        N: Integer (how many documents to retrieve). By default N = 3

        Returns:
        -----------
        a ranked list of pairs (doc_id, score) in the length of N.
        """

        return sorted([(doc_id, builtins.round(score, 5)) for doc_id, score in score_dict.items()], key=lambda x: x[1],
                      reverse=True)[:N]

    def get_candidate_documents(self, query_to_search, path, threshold):
        """
        Generate a dictionary representing a pool of candidate documents for a given query.

        Parameters:
        -----------
        query_to_search: list of tokens (str). This list will be preprocessed in advance (e.g., lower case, filtering stopwords, etc.').
                          Example: 'Hello, I love information retrival' --->  ['hello','love','information','retrieval']

        index:           inverted index loaded from the corresponding files.

        words,pls: generator for working with posting.
        Returns:
        -----------
        list of candidates. In the following format:
                                                                    key: pair (doc_id,term)
                                                                    value: tfidf score.
        """
        candidates = []
        c = Counter()
        for term in np.unique(query_to_search):
            if self.stem:
                term = self.lancaster.stem(term)
            if term in self.index.df:
                current_list = read_posting_list(self.index,term,path)[:100]
                candidates = [i[0] for i in current_list]
                c.update(candidates)
        tresh_num = int(len(query_to_search) * threshold)
        d = [x for x in c.keys() if c[x] >= tresh_num]
        return d

    def calc_idf(self, list_of_tokens):
        """
        This function calculate the idf values according to the BM25 idf formula for each term in the query.

        Parameters:
        -----------
        query: list of token representing the query. For example: ['look', 'blue', 'sky']

        Returns:
        -----------
        idf: dictionary of idf scores. As follows:
                                                    key: term
                                                    value: bm25 idf score
        """
        idf = {}
        for term in list_of_tokens:
            if self.stem:
                term = self.lancaster.stem(term)
            if term in self.index.df.keys():
                n_ti = self.index.df[term]
                idf[term] = math.log(1 + (self.N - n_ti + 0.5) / (n_ti + 0.5))
            else:
                pass
        return idf

    def calc_q_tf(self, query,path):
        total_terms = {}
        for term in query:
            if self.stem:
                term = self.lancaster.stem(term)
            if term in self.index.df:
                term_frequencies = dict(read_posting_list(self.index, term,path))
                total_terms[term] = term_frequencies
        return total_terms

    def search(self, query,path , threshold=0, N=100):
        """
        This function calculate the bm25 score for given query and document.
        We need to check only documents which are 'candidates' for a given query.
        This function return a dictionary of scores as the following:
                                                                    key: query_id
                                                                    value: a ranked list of pairs (doc_id, score) in the length of N.

        Parameters:
        -----------
        query: list of token representing the query. For example: ['look', 'blue', 'sky']
        doc_id: integer, document id.

        Returns:
        -----------
        score: float, bm25 score.
        """
        # YOUR CODE HERE
        if self.is_bi:
            temp = zip(*[query[i:] for i in range(0, 2)])
            query = [' '.join(terms) for terms in temp]
        dic_tf = self.calc_q_tf(query,path)
        self.idf = self.calc_idf(query)
        return self.get_top_n({relevant_doc: self._score(relevant_doc, dic_tf) for relevant_doc in
                               self.get_candidate_documents(query, path,threshold=threshold)}, N)

    def _score(self, doc_id, dic_tf):
        """
        This function calculate the bm25 score for given query and document.

        Parameters:
        -----------
        query: list of token representing the query. For example: ['look', 'blue', 'sky']
        doc_id: integer, document id.

        Returns:
        -----------
        score: float, bm25 score.
        """
        score = 0.0
        doc_len = self.index.NF[doc_id][1]
        for term in dic_tf:
            term_frequencies = dic_tf[term]
            if doc_id in term_frequencies.keys():
                freq = term_frequencies[doc_id]
                numerator = self.idf[term] * freq * (self.k1 + 1)
                denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / self.AVGDL)
                score += (numerator / denominator)
        return score


###### constant variables ######
TUPLE_SIZE = 6
TF_MASK = 2 ** 16 - 1  # Masking the 16 low bits of an integer
bucket_name = "proj_bucket_308013"
###### cosine similarity for search_body function ########

def cos_sim(query_to_search, index,path, threshold=0):
    ''' A function that compute the cosine similarity of a query to each relevent document
        in the corpus. relevent document is a document with at least 1 common word with the query.
    Parameters:
    -----------
      query_to_search: list of tokens.
      index: inverted index

    Returns:
    --------
      list of tuples (id, score) sorted bt score.
    '''
    query_to_search = [x.lower() for x in query_to_search]
    epsilon = .0000001
    docs = len(index.NF)
    sim = defaultdict(list)
    final_sim = {}
    query_terms = Counter(query_to_search)
    for token in np.unique(query_to_search):
        if token in index.df.keys():  # avoid terms that do not appear in the index.
            q_tf = query_terms[token] / len(query_to_search)
            df = index.df[token]
            idf = math.log((docs) / (df + epsilon), 2)
            q_tf_idf = q_tf * idf
            pl = read_posting_list(index, token, path)[:100]
            for d in pl:
                tf = d[1] / (index.NF[d[0]][1])
                tf_idf = tf * idf
                sim[d[0]].append(q_tf_idf * tf_idf)
    for k, v in sim.items():
        if len(v) / len(query_terms) >= threshold:
            valSum = builtins.sum(v)
            norm_q = 1 / math.sqrt(builtins.sum([t ** 2 for t in query_terms.values()]))
            norm = index.NF[k][0] * norm_q
            final_sim[k] = norm * valSum
    return sorted([(doc_id, score) for doc_id, score in final_sim.items()], key=lambda x: x[1], reverse=True)[:100]

##### function for search title and search anchor #####
def binary_rank(query_to_search,index,path):
    query_to_search = [x.lower() for x in query_to_search]
    rank = Counter()
    for token in np.unique(query_to_search):
        if token in index.df.keys(): #avoid terms that do not appear in the index.
            pl=read_posting_list(index,token,path)
            for d in pl:
                if path.find("title")!=-1:
                    if token in index.title_dict[d[0]].lower():
                        rank[d[0]]+=1
                else:
                    rank[d[0]] += 1
    return [(l,k) for k,l in sorted([(j,i) for i,j in rank.items()], reverse=True)]

##### get rank for list of docs id (for get pagerank and get views function #####
def get_rank_from_list(id_list,rank_dict,normalize=False,log_norm=False):
    if normalize:
        if log_norm:
            new_list = [math.log(rank_dict[i]+0.000001) for i in id_list]
        else:
            new_list = [rank_dict[i] for i in id_list]
        norm = builtins.max(new_list)
        return [x/norm for x in new_list]
    return [rank_dict[i] for i in id_list]


def merge_results(query, bm25_body, bm25_title, inverted_anchor, pagerank,views, N=100):
    """
    This function merge and sort documents retrieved by its weighte score (e.g., title and body).

    Parameters:
    -----------
    title_scores: a dictionary build upon the title index of queries and tuples representing scores as follows:
                                                                            key: query_id
                                                                            value: list of pairs in the following format:(doc_id,score)

    body_scores: a dictionary build upon the body/text index of queries and tuples representing scores as follows:
                                                                            key: query_id
                                                                            value: list of pairs in the following format:(doc_id,score)
    title_weight: float, for weigted average utilizing title and body scores
    text_weight: float, for weigted average utilizing title and body scores
    N: Integer. How many document to retrieve. This argument is passed to topN function. By default N = 3, for the topN function.

    Returns:
    -----------
    dictionary of querires and topN pairs as follows:
                                                        key: query_id
                                                        value: list of pairs in the following format:(doc_id,score).
    """
    # YOUR CODE HERE

    query = [x.lower() for x in query]

    title_weight = 0.55
    text_weight = 0.25
    anchor_weight = 0.2
    # bigram_title_weight = 0.1


    #
    body_scores = bm25_body.search(query,'postings_gcp/body_stem_index')
    title_scores = bm25_title.search(query, 'postings_gcp/title_index')
    anchor_scores = binary_rank(query,inverted_anchor,'postings_gcp/anchor_index')
    # bigram_score_title = bm25_bigram_title.search(query,'postings_gcp//body_stem_index')

    merges = {}
    title_d = defaultdict(lambda: 0, title_scores)
    body_d = defaultdict(lambda: 0, body_scores)
    anchor_d = defaultdict(lambda: 0, anchor_scores)
    # bigram_d = defaultdict(lambda: 0, bigram_score)
    # bigram_title_d = defaultdict(lambda: 0, bigram_score_title)

    # merge all doc ids that return from the differents indexes
    merge_keys = set(dict.fromkeys([k[0] for k in (title_scores + body_scores + anchor_scores )]))
    merge_keys = list(merge_keys)

    # norm pagerank and pageviews of the rellevants docs
    ranking_pr = get_rank_from_list(merge_keys,pagerank,normalize=True)
    norm_pagerank={merge_keys[i]:ranking_pr[i] for i in range(len(merge_keys))}

    ranking_v = get_rank_from_list(merge_keys, views, normalize=True, log_norm=True)
    norm_view = {merge_keys[i]: ranking_v[i] for i in range(len(merge_keys))}
    #
    #
    for k in merge_keys:
        merges[k] = (title_d[k] * title_weight + body_d[k] * text_weight + anchor_d[k] * anchor_weight)*((norm_pagerank[k]+norm_view[k])/2)
    merges = dict(sorted(merges.items(), key=lambda x: x[1], reverse=True))
    if N <= len(merges):
        merges = dict(list(merges.items())[:N])

    return  sorted(merges.items(), key=lambda x: x[1],reverse=True)

def norm_views (views):
    return { k: math.log(v) for k,v in views.items()}

def get_id_title(list,inverted_title):
    """
    :param list: list of tuples: [(doc_id,score)]
    :param inverted_title: inverted index of the title
    :return: list of tuples: [(doc_id, doc_title)]
    """
    return [(id,inverted_title.title_dict[id]) for id,x in list]

def read_posting_list(inverted, w, path):
    with closing(MultiFileReader()) as reader:
        locs = inverted.posting_locs[w]
        b = reader.read(locs, inverted.df[w] * TUPLE_SIZE,bucket_name,path)
        posting_list = []
        for i in range(inverted.df[w]):
            doc_id = int.from_bytes(b[i * TUPLE_SIZE:i * TUPLE_SIZE + 4], 'big')
            tf = int.from_bytes(b[i * TUPLE_SIZE + 4:(i + 1) * TUPLE_SIZE], 'big')
            if doc_id != 0:
                posting_list.append((doc_id, tf))
        return posting_list
