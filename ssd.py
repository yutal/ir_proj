import os

from flask import Flask, request, jsonify
from search_backend import *
from search_backend import BM25_from_index
import json
from google.cloud import storage
from inverted_index_gcp import MultiFileReader
from search_backend import *

os.environ['GOOGLE_APPLICATION_CREDENTIALS']=r'C:\Users\guyyu\retrivel-project-337416-0b0200d20959.json'

def load(base_dir, name):
    with open(Path(base_dir) / f'{name}.pkl', 'rb') as f:
        return pickle.load(f)

#
# view= load('C:\\Users\\guyyu\\Downloads','anchor_index')
# print(view.df['python'])
bucket_name = "proj_bucket_308013"
client = storage.Client()
blobs= client.list_blobs(bucket_name)
bucket = client.get_bucket(bucket_name)
# blob = bucket.get_blob('postings_gcp/anchor_index/pagerank_dict.pkl')
# for blob in client.list_blobs(bucket_name, prefix='postings_gcp/title_index'):
#   if not blob.name.endswith("pickle"):
#     continue
#   with blob.open("rb") as f:
#     data = pickle.load(f)
def read_pkl_buck(base_dir,name):
    for blob in client.list_blobs(bucket_name, prefix=base_dir):
        if not blob.name.endswith(f"{name}.pkl"):
            continue
        with blob.open("rb") as f:
            return pickle.load(f)

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

# pr= read_pkl_buck('postings_gcp/anchor_index','pagerank_dict')
# pagerank = title_d = defaultdict(lambda: 0, pr['pagerank'])
# print(math.log(min(pagerank.values())))
views= read_pkl_buck('postings_gcp/anchor_index','wid2pv')
views=  defaultdict(lambda: 0, views)
print(views[12772])
print((views[11002]))
merge_keys=[12772,11002]
ranking_pr = get_rank_from_list(merge_keys, views, normalize=True,log_norm=True)
norm_pagerank = {merge_keys[i]: ranking_pr[i] for i in range(len(merge_keys))}
print(norm_pagerank)
# ranking = get_rank_from_list(merge_keys,pagerank,normalize=True)
# norm_pagerank={merge_keys[i]:ranking[i] for i in range(len(merge_keys))}
# print(norm_pagerank)
# with blob.open("rb") as f:
#     data = pickle.load(f)
# def read_pkl_buck(base_dir,name):
#     for blob in client.list_blobs(bucket_name, prefix=base_dir):
#         if not blob.name.endswith(f"{name}.pkl"):
#             continue
#         with blob.open("rb") as f:
#             ver = pickle.load(f)
#     return ver

# inverted_body=read_pkl_buck('postings_gcp/body_stem_index','body_stem_index')
# print(inverted_body)
# print(len(inverted_body.NF))
# print(inverted_body.df["men"])
# print(inverted_body.df["python"])
# print(inverted_body.df["nipple"])
# bm25_title = BM25_from_index(index = inverted_body)
# title_scores=bm25_title.search(['nipples','men'],'postings_gcp/body_stem_index')
# print(title_scores)
# print(cos_sim(["men","nipples"],inverted_body,'postings_gcp/body_index'))
# print(dict(read_posting_list(inverted_body,"nipples",'postings_gcp/body_index')))