import os
import openai
import string
import requests
import json
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, Matern, DotProduct
from scipy.stats import ecdf, lognorm
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import random



article_writer = """Write an article on the topic listed bedlow

### TOPIC ###
{}
"""

article_rewriter = """You will be given a topic and an article. Rewrite the article so that it is more relevant to the topic.

### TOPIC ###
{}

### ARTICLE ###
{}
"""

OPENAI_MODEL = "gpt-4-mini"

def call_gpt(prompt, temp):

    client = openai.OpenAI(api_key=os.environ['OPENAI_KEY'])

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=temp
    )
    return response.choices[0].message.content




def rerank(topic, documents):
    try:
        response = requests.post(
          "https://api.cohere.com/v2/rerank",
          headers={
            "Authorization": "Bearer " + os.environ['COHERE_KEY']
          },
          json={
            "model": "rerank-v3.5",
            "query": topic,
            "documents": documents#,
    #        "top_n": 3
          },
        )
    except Exception as e:
        print(response)
        return json.loads({'x': 'error'})
    
    return response.json()




class Node:
    def __init__(self, parent, topic, text):
        self.parent = parent
        self.text = text
        self.topic = topic
        self.visits = 1
        self.score = -1
        self.relevance = None
        self.identifier = ''.join(random.choice(string.ascii_letters + string.digits) for i in range(10))
        self.children = []
        self.embedding = None


    def rollout(self, T):
        if T > 0:
            self.children.append(Node(self, self.topic, call_gpt(article_rewriter.format(self.topic, self.text), .9)))
            self.children[-1].rollout(T-1)

#    def expand(self):
#        self.children.append(Node(self, self.topic, call_gpt(article_rewriter.format(self.topic, self.txt), .9)))

    def get_rollout(self):
        if len(self.children) > 0:
            return [self.children[0]] + self.children[0].get_rollout()
        else:
            return []
        
    def get_scores(self):
        scores = [self.relevance]
        for c in self.children:
            scores += c.get_scores()
        return scores
    
    def get_eligible_nodes(self):
        
        if (len(self.get_scores()) > 2) & (len(self.children) < 10):
            eligible = [self]
            for c in self.children:
                eligible += c.get_eligible_nodes()
            return eligible
        else:
            return []

    



    
class qEI:
    def __init__(self, embeddings):
        self.embeddings = embeddings

    def fit(self, scores):
        self.gpr = GaussianProcessRegressor(kernel = Matern() + WhiteKernel())
        scores_ecdf = ecdf(scores)
        transformed_scores = np.log(lognorm.ppf(scores_ecdf.cdf.evaluate(scores) * .999 + .0005, 1))
        self.gpr.fit(self.embeddings, transformed_scores)
        self.y_best = max(transformed_scores)
        return self.y_best

        
    def create_batches(self, n_batches, batch_size):
        self.batch_size = batch_size
        self.batch_mu = []
        self.batch_sigma = []
    
        self.batches = []
        self.batch_idx = []
        n_to_choose_from = len(self.embeddings)
        for z in range(n_batches):
            batch = []
            for x in range(batch_size):
                rx = random.randint(0, n_to_choose_from-1)
                while rx in batch:
                    rx = random.randint(0, n_to_choose_from-1)
                batch.append(rx)
            self.batch_idx.append(batch)
            m, s = self.gpr.predict([self.embeddings[i] for i in batch], return_cov=True)
            self.batch_mu.append(','.join([str(x) for x in m]))
            self.sigma = []
            for x in s:
                self.sigma.append(','.join([str(y) for y in x]))
            self.batch_sigma.append(';'.join(self.sigma))
        
        
    def get_best_batch(self, gpu=False):
        try:
            if gpu:
                url = f'http://34.130.49.1:5000/gpu_qei'
            else:
                url = f'https://boaz.onrender.com/qei'
#                url = f'http://3.132.240.115:8080/qei'
            data = {'y_best': str(self.y_best),
                    'n': str(self.batch_size),
                    'k': ';'.join(self.batch_mu),
                    'sigma': '|'.join(self.batch_sigma)}
            headers = {"Content-Type": "application/json",
                       "Accept": "application/json",
                       "X-RapidAPI-Key": os.environ['X_RapidAPI_KEY']
                      }
            response = requests.post(url,json.dumps(data), headers=headers)
            boaz = eval(response.content.decode('utf-8'))
        except Exception as e:
            print('Bayesian Issues:', e)
            return random.randint(0, len(self.batch_mu))
        fboaz = [float(x) for x in boaz['scores'].split(',')]
        best = -1
        
        for i, mx in enumerate(fboaz):
            if mx > best:
                best = float(mx)
                best_idx = i
        return self.batch_idx[best_idx]

def initial_Tree(topic, n_parallel):
    articles = []

    with ThreadPoolExecutor(max_workers=n_parallel) as executor:
        futures = [executor.submit(call_gpt, article_writer.format(topic), .9) for t in range(n_parallel)]    
        for future in futures:
            articles.append(future.result())
    # initial Tree
    Tree = []
    for x in articles:
        Tree.append(Node(None, topic, x))
    
    # Initial ranks
    # Do as a batch because the trial key is throttled
    raw_rank = rerank(topic, [t.text for t in Tree])
    ranks = [r['relevance_score'] for r in raw_rank['results']]
    for t, r in zip(Tree, ranks):
        t.relevance = r

    
    client = openai.OpenAI(api_key=os.environ['OPENAI_KEY'])

    #initial embeddings
    embeddings = client.embeddings.create(
        input=[t.text for t in Tree],
        model="text-embedding-3-small"
    ).data
    for t, e in zip(Tree, embeddings):
        t.embedding = e.embedding

    return Tree



def roll_and_rank(N, n):
    N.rollout(n)
    rolls = [N] + N.get_rollout()
    roll_text = [N.text] + [x.text for x in rolls]
    raw_rank = rerank(N.topic, roll_text)
    try:
        ranks = [x['relevance_score'] for x in raw_rank['results']]
    except Exception as e:
        print(e)
        print(raw_ranks)
    for r, n in zip(ranks, rolls):
        n.relevance = r

    client = openai.OpenAI(api_key=os.environ['OPENAI_KEY'])

    embeddings = client.embeddings.create(
        input=roll_text,
        model="text-embedding-3-small"
    ).data
    
    for t, e in zip(rolls, embeddings):
        t.embedding = e.embedding
        

    return N.relevance

def roll_and_rank_tree(Tree, depth):
    results = []
    with ThreadPoolExecutor(max_workers=len(Tree)) as executor:
        futures = [executor.submit(roll_and_rank, t, depth) for t in Tree]    
        for future in futures:
            results.append(future.result() )
    return results

def get_all(t):
    all_nodes = [t]
    for c in t.children:
        all_nodes += get_all(c)
    return all_nodes

def get_batch_from_tree(Tree, batch_size, n_batches):
    eligible = []
    for t in Tree:
        eligible += t.get_eligible_nodes()
    
    scores = []
    embeddings = []
    eligibles = []
    for e in eligible:
        s = e.get_scores()
        scores += s
        embeddings += [e.embedding] * len(s)
        if not e.embedding:
            return "ERROR"
        eligibles += [e] * len(s)
    chooser = qEI(embeddings)
    y_best = chooser.fit(scores)
    chooser.create_batches(n_batches, batch_size)
    return chooser.get_best_batch(), eligibles
    


def expand(N, ix):
    N.children.append(Node(N, N.topic, call_gpt(article_rewriter.format(N.topic, N.text), .9)))    
    roll_and_rank(N.children[-1], 3)
    return N.relevance

def iterate2(batch_size, eligibles, next_batch):
    
    start = datetime.now()
    result = []
    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        futures = [executor.submit(expand, eligibles[node_id], node_id) for node_id in next_batch]    
        for future in futures:
            result.append(future.result() )
    return (datetime.now() - start).seconds



def iterate(Tree, batch_size, n_batches):

    eligible = []
    for t in Tree:
        eligible += t.get_eligible_nodes()
    
    scores = []
    embeddings = []
    eligibles = []
    for e in eligible:
        s = e.get_scores()
        scores += s
        embeddings += [e.embedding] * len(s)
        if not e.embedding:
            return "ERROR"
        eligibles += [e] * len(s)
    chooser = qEI(embeddings)
    y_best = chooser.fit(scores)
    chooser.create_batches(n_batches, batch_size)
    next_batch = chooser.get_best_batch()
    print(next_batch) 
    start = datetime.now()
    result = []
    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        futures = [executor.submit(expand, eligibles[node_id], node_id) for node_id in next_batch]    
        for future in futures:
            result.append(future.result() )
            
    return (datetime.now() - start).seconds


def get_all(t):
    all_nodes = [t]
    for c in t.children:
        all_nodes += get_all(c)
    return all_nodes



def best_worst(Tree):
    all_nodes = []
    best = -1
    worst = 2
    for t in Tree:
        all_nodes += get_all(t)
    for n in all_nodes:
        if n.relevance > best:
            best = n.relevance
            best_N = n
        if n.relevance < worst:
            worst = n.relevance
            worst_N = n
    return best_N, worst_N
