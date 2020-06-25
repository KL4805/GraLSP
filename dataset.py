import networkx as nx 
import numpy as np
import random
from six.moves import xrange 
import math
import time
import copy

class dataset:
    def __init__(self, dataset_name = "cora", purpose = "link_prediction", linkpred_ratio = 0.1, 
         path_length=15, num_paths=50, 
         window_size=3, batch_size=100, neg_size=5, num_skips = 5, num_neighbor = 20, 
         anonym_walk_len = 6, p = 0.25, q = 1):

        assert purpose in ["classification", "link_prediction", "none"]
        assert batch_size % num_skips == 0

        self.dataset_name = dataset_name
        self.path_length = path_length
        self.num_paths = num_paths
        self.window_size = window_size
        self.batch_size = batch_size
        self.neg_size = neg_size
        self.num_skips = num_skips
        self.data_idx = 0
        self.anonym_walk_len = anonym_walk_len
        self.purpose = purpose
        self.linkpred_ratio = linkpred_ratio
        self.p = p
        self.q = q

        self.edge_list = []
        with open("data/" + dataset_name + "/edges", 'r') as infile:
            for line in infile:
                elems = line.rstrip().split(' ')
                src, dst = int(elems[0]), int(elems[1])
                if src == dst:
                    continue
                self.edge_list.append((src, dst))
        self.num_nodes = 1 + max(max([u[0] for u in self.edge_list]), max([u[1] for u in self.edge_list]))

        if self.purpose != "link_prediction":
            # construct networkx graph, with all edges
            self.g = nx.Graph()
            for i in range(self.num_nodes):
                self.g.add_node(i)
            for i, j in self.edge_list:
                self.g.add_edge(i, j)
        else:
            # with link prediction, no need for labels, but we need to out sample a few links
            self.degree_seq = np.zeros((self.num_nodes))
            for s, d in self.edge_list:
                self.degree_seq[s] += 1
                self.degree_seq[d] += 1

            # sample prediction links, ensures that every node has at least one link in training
            cnt = 0
            to_be_sampled = int(self.linkpred_ratio * len(self.edge_list))
            self.pred_edge_list = []
            while cnt < to_be_sampled:
                s, d = random.choice(self.edge_list)
                if self.degree_seq[s] == 1 or self.degree_seq[d] == 1:
                    continue
                else:
                    self.degree_seq[s] -= 1
                    self.degree_seq[d] -= 1
                    self.pred_edge_list.append((s, d))
                    self.edge_list.remove((s, d))
                    cnt += 1

            self.g = nx.Graph()
            for i in range(self.num_nodes):
                self.g.add_node(i)
            for i, j in self.edge_list:
                self.g.add_edge(i, j)  

            

        degree_seq_dict = dict(self.g.degree)
        self.degree_seq = [degree_seq_dict[i] for i in range(self.num_nodes)]
        self.neg_sampling_seq = []
        self.preprocess_transition_prob()
        self.random_walks = []

        nodes = list(range(self.num_nodes))
        for _ in range(self.num_paths):
            random.shuffle(nodes)
            for node in nodes:
                #walk = self.generate_random_walk(node, self.path_length)
                walk = self.node2vec_walk(node, self.path_length)
                self.random_walks.append(walk)
            
        self.node_walks = [[] for i in range(self.num_nodes)]
        for w in self.random_walks:
            self.node_walks[w[0]].append(w)
        self.node_anonymous_walks = [[] for i in range(self.num_nodes)]
        self.node_walk_radius = [[] for i in range(self.num_nodes)]
        for ws in range(self.num_nodes):
            for w in self.node_walks[ws]:
                self.node_anonymous_walks[ws].append(self.to_anonym_walk(w))
                self.node_walk_radius[ws].append(int(2*self.anonym_walk_len/len(np.unique(w[:10]))))
                
        self.types_and_nodes = [[] for i in range(self.num_nodes)]
        
        
        self.node_walks = np.array(self.node_walks).astype(int)
        self.node_anonym_walktypes = np.zeros((self.num_nodes, self.num_paths))
        self.node_normalized_walk_distr = self.process_anonym_distr(self.anonym_walk_len)
        self.anonym_walk_dim = len(self.node_normalized_walk_distr[0])
        
        for ws in range(self.num_nodes):
            for _ in range(self.num_paths * self.num_skips):
                wk = random.randint(0, self.num_paths-1)
                self.types_and_nodes[ws].append([self.node_anonym_walktypes[ws][wk].astype(int), random.choice(self.node_walks[ws][wk][:self.node_walk_radius[ws][wk]])])
        self.types_and_nodes = np.array(self.types_and_nodes).astype(int)

        
        self.node_features = np.load("data/" + dataset_name + "/features.npy")
        self.feature_dim = len(self.node_features[0])
        
        if self.purpose == "classification":
            self.node2label = np.zeros((self.num_nodes))

            with open("data/" + dataset_name + "/node2label") as infile:
                for line in infile:
                    elems = line.rstrip().split(" ")
                    node, label = int(elems[0]), int(elems[1])
                    self.node2label[node] = label
            self.node2label = self.node2label.astype(int)



        for i in range(self.num_nodes):
            distr = math.pow(self.degree_seq[i], 0.75)
            distr = math.ceil(distr)
            for _ in range(distr):
                self.neg_sampling_seq.append(i)
            # create adj_info used in graphsage sampling
        self.adj_info = np.zeros((int(self.num_nodes), int(max(self.degree_seq))))
        self.max_degree = max(self.degree_seq)
        for node in range(self.num_nodes):
            neighbors = self.get_neighbor(node)
            if len(neighbors) < self.max_degree:
                neighbors = np.random.choice(neighbors, int(self.max_degree), replace = True)
            self.adj_info[node] = neighbors
        self.adj_info = self.adj_info.astype(int)
    def get_neighbor(self, node):
        # return neighbor node set of a certain node
        neighbor = [n for n in self.g.neighbors(node)]
        return neighbor         
    
    def get_alias_edge(self, src, dst):
        unnormalized_probs = []
        for dst_nbr in sorted(self.g.neighbors(dst)):
            if dst_nbr == src:
                unnormalized_probs.append(1/self.p)
            elif self.g.has_edge(dst_nbr, src):
                # one hop neighbor
                unnormalized_probs.append(1)
            else:
                unnormalized_probs.append(1/self.q)
        normalize_const = np.sum(unnormalized_probs)
        normalized_probs = [prob/normalize_const for prob in unnormalized_probs]
        return alias_setup(normalized_probs)
    
    def preprocess_transition_prob(self):
        alias_nodes = {}
        for nodes in self.g.nodes():
            normalized_probs = [1/self.degree_seq[nodes] for i in range(self.degree_seq[nodes])]
            alias_nodes[nodes] = alias_setup(normalized_probs)
            
        alias_edges = {}
        
        for edge in self.g.edges():
            alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
            alias_edges[(edge[1], edge[0])] = self.get_alias_edge(edge[1], edge[0])
        
        self.alias_nodes = alias_nodes
        self.alias_edges = alias_edges
        
        return
                
    
    def generate_random_walk(self, begin_node, path_length):
        path = [begin_node]
        while(len(path) < path_length):
            candidates = list(dict(self.g[path[-1]]).keys())
            nextnode = random.choice(candidates)
            path.append(nextnode)
        return path
    
    def node2vec_walk(self, begin_node, path_length):
        walk = [begin_node]
        alias_nodes = self.alias_nodes
        alias_edges = self.alias_edges
        
        while(len(walk) < path_length):
            cur = walk[-1]
            cur_neighbors = self.get_neighbor(cur)
            cur_neighbors = sorted(cur_neighbors)
            if len(cur_neighbors):
                if len(walk) == 1:

                    abc = alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])

                    walk.append(cur_neighbors[abc])
                else:
                    prev = walk[-2]
                    nextnode = cur_neighbors[alias_draw(alias_edges[(prev, cur)][0], 
                        alias_edges[(prev, cur)][1])]
                    walk.append(nextnode)
            else:
                break

        return walk
            

    def generate_batch(self, batch_size, window_size):
        keys, labels =[], []
        i = 0
        while i < batch_size:
            #for j in range(self.num_skips):
            thiskey = self.random_walks[self.data_idx][0]
            thislabel = self.random_walks[self.data_idx][random.randint(1, self.window_size-1)]
                    
            keys.append(thiskey)
            labels.append(thislabel)
            self.data_idx += 1
            self.data_idx %= len(self.random_walks)
            i +=1
        
        negs = self.negative_sampling(keys, labels, self.neg_size)


        walk_key, walk_label, walk_neg = self.generate_walk_prox(keys)
        return np.array(keys).astype(int), np.array(labels).astype(int), np.array(negs).astype(int), np.array(walk_key).astype(int), np.array(walk_label).astype(int), np.array(walk_neg).astype(int)


    def negative_sampling(self, keys, labels, neg_size):
        negs = np.zeros((neg_size))

        for j in range(neg_size):
            neg_ = random.choice(self.neg_sampling_seq)
            while (neg_ in labels or neg_ in keys):
                neg_ = random.choice(self.neg_sampling_seq)
            negs[j] = neg_
        return negs            

    def generate_walk_prox(self, batch):
        # for each node, generate three walks i, j, k, pi*pj > 0, pi*pk < 0
        walki, walkj, walkk = [], [], []
        for node in batch:
            #print(node)
            for _ in range(2):
                while 1:
                    i = random.choice(list(range(len(self.node_normalized_walk_distr[0]))))
                    popi = self.node_normalized_walk_distr[node][i]
                    if popi == 0:
                        continue
                    else:
                        break
                positive = -1
                negative = -1
                while 1:
                    j = random.choice(list(range(len(self.node_normalized_walk_distr[0]))))
                    if positive < 0 and self.node_normalized_walk_distr[node][j]*popi > 0:
                        positive = j
                    elif negative < 0 and self.node_normalized_walk_distr[node][j]*popi < 0:
                        negative = j
                    if positive>=0 and negative >=0:
                        break
                walki.append(i)
                walkj.append(positive)
                walkk.append(negative)
        return walki, walkj, walkk

    def to_anonym_walk(self, walk):
        # convert a walk sequence to an anonymous walk
        num_app = 0
        apped = dict()
        anonym = []
        for node in walk:
            if node not in apped:
                num_app += 1
                apped[node] = num_app
            anonym.append(apped[node])

        return anonym


    def process_anonym_distr(self, length):
        # process anonymous walks of length= length
        self.anonym_walk_dict = generate_walk2num_dict(length)
        node_anonym_distr = np.zeros((self.num_nodes, len(self.anonym_walk_dict)))
        for n in range(self.num_nodes):
            for idxw in range(len(self.node_anonymous_walks[n])):
                w = self.node_anonymous_walks[n][idxw]
                strw = intlist_to_str(w[:length])
                wtype = self.anonym_walk_dict[strw]
                self.node_anonym_walktypes[n][idxw] = wtype
                node_anonym_distr[n][wtype] += 1
        node_anonym_distr /=self.num_paths
        self.graph_anonym_distr = np.mean(node_anonym_distr, axis = 0)
        graph_anonym_std = np.std(node_anonym_distr, axis = 0)
        graph_anonym_std[np.where(graph_anonym_std == 0)] = 0.001
        # In case divided by 0
        #print(graph_anonym_distr[np.where(graph_anonym_std==0)])
        return (node_anonym_distr - self.graph_anonym_distr)/graph_anonym_std

    def sample_negative_links(self, negative_num):
        assert self.purpose == "link_prediction"
        negative_edges = []
        cnt = 0
        while cnt < negative_num:
            s = random.choice(list(range(self.num_nodes)))
            d = random.choice(list(range(self.num_nodes)))
            if (s, d) in self.edge_list or (s, d) in self.pred_edge_list or (d, s) in self.edge_list or (d, s) in self.pred_edge_list:
                continue
            negative_edges.append((s, d))
            cnt += 1
        return negative_edges


            








def generate_anonym_walks(length):
    anonymous_walks = []
    def generate_anonymous_walk(totlen, pre):
        if len(pre) == totlen:
            anonymous_walks.append(pre)
            return
        else:
            candidate = max(pre) + 1
            for i in range(1, candidate+1):
                if i!= pre[-1]:
                    npre = copy.deepcopy(pre)
                    npre.append(i)
                    generate_anonymous_walk(totlen, npre)
    generate_anonymous_walk(length, [1])
    return anonymous_walks

def generate_walk2num_dict(length):
    anonym_walks = generate_anonym_walks(length)
    anonym_dict = dict()
    curid = 0
    for walk in anonym_walks:
        swalk = intlist_to_str(walk)
        anonym_dict[swalk] = curid
        curid += 1
    # verified that anonym_walks won't have duplicate results
    return anonym_dict

def intlist_to_str(lst):
    slst = [str(x) for x in lst]
    strlst = "".join(slst)
    return strlst


def alias_setup(probs):
    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K).astype(int)
    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        q[kk] = K * prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)
    
    while len(smaller) >0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()
        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)
    
    return J, q

def alias_draw(J, q):
    K = len(J)
    
    kk = int(np.floor(np.random.rand()*K))
    if np.random.rand()<q[kk]:
        return kk
    else:
        return J[kk]