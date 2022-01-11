import pickle
from flask import Flask, request, jsonify
from search_backend import *
from search_backend import BM25_from_index
from inverted_index_gcp import *
from google.cloud import storage

class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)

bucket_name = "proj_bucket_308013"
client = storage.Client()

def load(base_dir, name):
    with open(Path(base_dir) / f'{name}.pkl', 'rb') as f:
        return pickle.load(f)

def read_pkl_buck(base_dir,name):
    for blob in client.list_blobs(bucket_name, prefix=base_dir):
        if not blob.name.endswith(f"{name}.pkl"):
            continue
        with blob.open("rb") as f:
            return pickle.load(f)


# pagerank= load('/home/guy_yutal/postings_gcp/anchor_index','pagerank_dict')
# views=load('/home/guy_yutal/postings_gcp/anchor_index','wid2pv')
# inverted_body=load('/home/guy_yutal/postings_gcp/body_index','body_index')
# inverted_title=load('/home/guy_yutal/postings_gcp/title_index','title_index')
# inverted_anchor=lsoad('/home/guy_yutal/postings_gcp/anchor_index','anchor_index')
# inverted_bigram_title=load('/home/guy_yutal/postings_gcp/bigram_title_index','bigram_title_index')
# inverted_body_stem=load('/home/guy_yutal/postings_gcp/body_stem_index','body_stem_index')

pr= read_pkl_buck('postings_gcp/anchor_index','pagerank_dict')
pagerank = title_d = defaultdict(lambda: 0, pr['pagerank'])
views= read_pkl_buck('postings_gcp/anchor_index','wid2pv')
inverted_body=read_pkl_buck('postings_gcp/body_index','body_index')
inverted_title=read_pkl_buck('postings_gcp/title_index','title_index')
inverted_anchor=read_pkl_buck('postings_gcp/anchor_index','anchor_index')
# inverted_bigram_title=read_pkl_buck('postings_gcp/bigram_title_index','bigram_title_index')
inverted_body_stem=read_pkl_buck('postings_gcp/body_stem_index','body_stem_index')

# pagerank= InvertedIndex.read_index('.','pagerank_dict')
# views=InvertedIndex.read_index('.','wid2pv')
# # inverted_body=InvertedIndex.read_index('.','body_index')
# inverted_title=InvertedIndex.read_index('.','title_index')
# inverted_anchor=InvertedIndex.read_index('.','anchor_index')
# # inverted_bigram_title=InvertedIndex.read_index('.','bigram_title_index')
# inverted_body_stem=InvertedIndex.read_index('.','body_stem_index')


bm25_body = BM25_from_index(index = inverted_body,stem=False)
bm25_title = BM25_from_index(index = inverted_title)
# bm25_anchor = BM25_from_index(index = inverted_anchor)
# bm25_bigram_title = BM25_from_index(index = inverted_bigram_title,is_bi=True)
bm25_body_stem = BM25_from_index(index = inverted_body_stem,stem=True)

app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False


@app.route("/search")
def search():
    ''' Returns up to a 100 of your best search results for the query. This is 
        the place to put forward your best search engine, and you are free to
        implement the retrieval whoever you'd like within the bound of the 
        project requirements (efficiency, quality, etc.). That means it is up to
        you to decide on whether to use stemming, remove stopwords, use 
        PageRank, query expansion, etc.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    query = query.split()
    score_list=merge_results(query, bm25_body_stem, bm25_title, inverted_anchor,pagerank,views)
    res=get_id_title(score_list,inverted_title)
    # END SOLUTION
    return jsonify(res)

@app.route("/search_body")
def search_body():
    ''' Returns up to a 100 search results for the query using TFIDF AND COSINE
        SIMILARITY OF THE BODY OF ARTICLES ONLY. DO NOT use stemming. DO USE the 
        staff-provided tokenizer from Assignment 3 (GCP part) to do the 
        tokenization and remove stopwords. 

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search_body?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    query = query.split()
    score_list=cos_sim(query, inverted_body,'postings_gcp/body_index', threshold=0)
    res = get_id_title(score_list, inverted_title)
    # END SOLUTION
    return jsonify(res)

@app.route("/search_title")
def search_title():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE TITLE of articles, ordered in descending order of the NUMBER OF 
        QUERY WORDS that appear in the title. For example, a document with a 
        title that matches two of the query words will be ranked before a 
        document with a title that matches only one query term. 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_title?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    query = query.split()
    score_list=binary_rank(query, inverted_title,'postings_gcp/title_index')
    res = get_id_title(score_list, inverted_title)
    # END SOLUTION
    return jsonify(res)

@app.route("/search_anchor")
def search_anchor():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE ANCHOR TEXT of articles, ordered in descending order of the 
        NUMBER OF QUERY WORDS that appear in anchor text linking to the page. 
        For example, a document with a anchor text that matches two of the 
        query words will be ranked before a document with anchor text that 
        matches only one query term. 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_anchor?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    query=query.split()
    score_list=binary_rank(query, inverted_anchor,'postings_gcp/anchor_index')
    res = get_id_title(score_list, inverted_title)
    # END SOLUTION
    return jsonify(res)

@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():
    ''' Returns PageRank values for a list of provided wiki article IDs. 

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pagerank
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pagerank', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of floats:
          list of PageRank scores that correrspond to the provided article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    res=get_rank_from_list(wiki_ids,pagerank)
    # END SOLUTION
    return jsonify(res)

@app.route("/get_pageview", methods=['POST'])
def get_pageview():
    ''' Returns the number of page views that each of the provide wiki articles
        had in August 2021.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pageview
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pageview', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ints:
          list of page view numbers from August 2021 that correrspond to the 
          provided list article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    res = get_rank_from_list(wiki_ids,views)
    # END SOLUTION
    return jsonify(res)


if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=True)
