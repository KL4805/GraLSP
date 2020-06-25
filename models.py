import tensorflow as tf 
import numpy as np
from dataset import *
import networkx as nx 
from six.moves import xrange
import math
import time
import threading
import queue
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import cross_val_score
import os


class GraLSP:
    def __init__(self, sess, dataset_name = "cora", purpose = "classification", linkpred_ratio = 0.1, 
        path_length = 10, num_paths = 100, window_size = 4, batch_size = 100, neg_size = 8, learning_rate = 0.005,
        optimizer = "Adam", embedding_dims = 30, save_path = "embeddings/GraLSP", num_steps = 10000, num_skips = 5, 
        hidden_dim = 100, num_neighbor = 40, anonym_walk_len = 8, walk_loss_lambda = 0.1, 
        walk_dim = 30, p = 0.25, q = 1):

        self.sess = sess
        self.batch_size = batch_size
        self.dataset_name = dataset_name
        self.path_length = path_length
        self.num_paths = num_paths
        self.window_size = window_size
        self.neg_size = neg_size
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.embedding_dims = embedding_dims
        self.save_path = save_path + "/" + self.dataset_name
        self.num_steps = num_steps
        self.num_skips = num_skips
        self.hidden_dim = hidden_dim
        self.num_neighbor = num_neighbor
        self.anonym_walk_len = anonym_walk_len
        self.walk_loss_lambda = walk_loss_lambda
        self.purpose = purpose
        self.linkpred_ratio = linkpred_ratio
        self.walk_dim = walk_dim
        self.p = p
        self.q = q

        self.start_time = time.time()
        if not os.path.exists(self.save_path + "/" + str(self.embedding_dims)):
            os.makedirs(self.save_path + "/" + str(self.embedding_dims))

        self.Dataset = dataset(self.dataset_name, self.purpose, self.linkpred_ratio, 
            self.path_length, self.num_paths, self.window_size, self.batch_size, self.neg_size, 
            self.num_skips, anonym_walk_len=self.anonym_walk_len, p = self.p, q = self.q)
        
        self.num_nodes = self.Dataset.num_nodes
        self.feature_dims = len(self.Dataset.node_features[0])
        self.node_features = self.Dataset.node_features
        
        self.num_anonym_walk_types = len(self.Dataset.node_normalized_walk_distr[0])
        print(self.num_anonym_walk_types)
        print(self.Dataset.node_anonym_walktypes.shape)
        self.build_model()
        print("[%.2fs]Finish sampling, begin training"%(time.time()-self.start_time))

        self.saver = tf.train.Saver()
    
    def build_model(self):
        """
            the model takes in: 
            walk: key, label, neg
            node: key, label, neg
            walks are dealt with directly while nodes will need sampling
        """
        self.neighs_and_types = self.Dataset.types_and_nodes
        print(np.max(self.neighs_and_types[:, :, 0]))
        print(np.max(self.neighs_and_types[:, :, 1]))

        self.batch_keys = tf.placeholder(tf.int32, [None])
        self.batch_labels = tf.placeholder(tf.int32, [None])
        self.batch_negs = tf.placeholder(tf.int32, [None])
        self.batch_input = tf.placeholder(tf.int32, [None])
        self.input_size = tf.placeholder(tf.int32)

        self.key_walks = tf.placeholder(tf.int32, [None])
        self.label_walks = tf.placeholder(tf.int32, [None])
        self.neg_walks = tf.placeholder(tf.int32, [None])

        self.nodes_keys, self.paths_keys = self.sample(self.batch_keys, self.num_neighbor, self.batch_size)
        self.nodes_labels, self.paths_labels = self.sample(self.batch_labels, self.num_neighbor, self.batch_size)
        self.nodes_negs, self.paths_negs = self.sample(self.batch_negs, self.num_neighbor, self.neg_size)
        self.nodes_inputs, self.paths_inputs = self.sample(self.batch_input, self.num_neighbor, self.input_size)

        self.walk_embeddings = tf.get_variable("walk_embeddings", [self.num_anonym_walk_types, self.walk_dim], tf.float64, 
            initializer = tf.contrib.layers.xavier_initializer())
        self.walk_loss = self.compute_walk_loss()

        self.output_keys = self.aggregate(self.nodes_keys, self.paths_keys, self.batch_size)#, compute_regularizer = True)
        self.output_labels = self.aggregate(self.nodes_labels, self.paths_labels, self.batch_size)
        self.output_negs = self.aggregate(self.nodes_negs, self.paths_negs, self.neg_size)
        self.output = self.aggregate(self.nodes_inputs, self.paths_inputs, self.input_size)

        self.output_keys = tf.nn.l2_normalize(self.output_keys, 1)
        self.output_labels = tf.nn.l2_normalize(self.output_labels, 1)
        self.output_negs = tf.nn.l2_normalize(self.output_negs, 1)
        self.output = tf.nn.l2_normalize(self.output, 1)

        pos_aff = tf.reduce_sum(tf.multiply(self.output_keys, self.output_labels), axis = 1)
        neg_aff = tf.einsum("ij,kj->ik", self.output_keys, self.output_negs)
        self.likelihood = tf.log(tf.sigmoid(pos_aff) + 1e-6) + tf.reduce_sum(tf.log(1-tf.sigmoid(neg_aff) + 1e-6), axis =1 )
        
        self.link_loss = -tf.reduce_mean(self.likelihood)
        self.walk_loss *= self.walk_loss_lambda
        self.loss = self.link_loss + self.walk_loss
        #self.loss += self.regu_lambda * self.l2_loss
        
        if self.optimizer == "Adam":
            self.optim = tf.train.AdamOptimizer(self.learning_rate)
        elif self.optimizer == "SGD":
            self.optim = tf.train.GradientDescentOptimizer(self.learning_rate)
        elif self.optimizer == "Momentum":
            self.optim = tf.train.MomentumOptimizer(learning_rate= self.learning_rate, momentum = 0.9)
        
        # Clipping
        # grads_and_vars = self.optim.compute_gradients(self.loss)
        # clipped_grads_and_vars = [(tf.clip_by_value(grad, -5.0, 5.0) if grad is not None else None, var)
        #     for grad, var in grads_and_vars]
        # self.opt_op = self.optim.apply_gradients(clipped_grads_and_vars)

        # No clipping
        self.opt_op = self.optim.minimize(self.loss)
            
    def compute_walk_loss(self):
        # self.walk_embeddings = tf.nn.l2_normalize(self.walk_embeddings, 1)
        walk_key_embed = tf.nn.embedding_lookup(self.walk_embeddings, self.key_walks)
        walk_label_embed = tf.nn.embedding_lookup(self.walk_embeddings, self.label_walks)
        walk_neg_embed = tf.nn.embedding_lookup(self.walk_embeddings, self.neg_walks)
        u_ijk = tf.reduce_sum(walk_key_embed*(walk_label_embed - walk_neg_embed), axis = 1)
        walk_loss = -tf.reduce_mean(tf.log(tf.sigmoid(u_ijk)))
        return walk_loss


    def sampleNeighborPath(self, batch_nodes, num_samples):
        adj_lists = tf.nn.embedding_lookup(self.neighs_and_types, batch_nodes)
        adj_lists = tf.transpose(tf.random_shuffle(tf.transpose(adj_lists, perm = [1, 0, 2])), perm = [1, 0, 2])
        adj_lists = tf.slice(adj_lists, [0, 0, 0], [-1, num_samples, -1])
        path_types = tf.slice(adj_lists, [0, 0, 0], [-1, -1, 1])
        neigh_nodes = tf.slice(adj_lists, [0, 0, 1], [-1, -1, 1])
        print(path_types.get_shape())
        print(neigh_nodes.get_shape())
        return tf.squeeze(path_types), tf.squeeze(neigh_nodes)
    
    def sample(self, inputs, num_sample, input_size):
        samples = [inputs]
        paths = []
        support_size = input_size
        for k in range(2):
            support_size *= num_sample
            sample_paths, nodes = self.sampleNeighborPath(samples[k], num_sample)
            samples.append(tf.reshape(nodes, [support_size]))
            paths.append(tf.reshape(sample_paths, [support_size]))
        return samples, paths

    def aggregate(self, sample_nodes, sample_paths, input_size):#, compute_regularizer = False):
        dims = [self.feature_dims, self.hidden_dim, self.embedding_dims]
        support_sizes = [1, self.num_neighbor, self.num_neighbor**2]
        hidden_nodes = [tf.nn.embedding_lookup(self.node_features, nodes) for nodes in sample_nodes]
        hidden_paths = [tf.nn.embedding_lookup(self.walk_embeddings, paths) for paths in sample_paths]
        #hidden_nodes_struct = [tf.nn.embedding_lookup(self.walk_embedded_nodes, nodes) for node in sample_nodes]
        #l2_losses = []
        for layer in range(2):
            input_dim = dims[layer]
            output_dim = dims[layer + 1]
            with tf.variable_scope("aggregator_" + str(layer), reuse = tf.AUTO_REUSE):
                weight_self = tf.get_variable("weight_self", [input_dim, output_dim], dtype = tf.float64, 
                    initializer = tf.contrib.layers.xavier_initializer())
                weight_neigh = tf.get_variable("weight_neigh", [input_dim, output_dim], dtype = tf.float64, 
                    initializer = tf.contrib.layers.xavier_initializer())
                weight_path = tf.get_variable("weight_path", [self.walk_dim, input_dim], dtype = tf.float64, 
                    initializer = tf.contrib.layers.xavier_initializer())
                '''if compute_regularizer:
                    l2_losses.append(tf.nn.l2_loss(weight_self))
                    l2_losses.append(tf.nn.l2_loss(weight_neigh))
                    l2_losses.append(tf.nn.l2_loss(weight_path))'''
                bias_path = tf.get_variable("bias_path", [input_dim], dtype = tf.float64, initializer = tf.constant_initializer(0.01))
                bias_aggregate = tf.get_variable("bias_aggregate", [output_dim], dtype = tf.float64, initializer = tf.constant_initializer(0.01))
                #bias_attention = tf.get_variable("bias_attention", dtype = tf.float64, initializer = tf.constant_initializer(0.01))
                #weight_attention = tf.get_variable("bias_weight", [self.walk_dim], dtype = tf.float64, initializer = tf.contrib.layers.xavier_initializer())
                next_hidden = []
                for hop in range(2-layer):
                    neigh_node_dims = [input_size * support_sizes[hop], self.num_neighbor, dims[layer]]
                    neigh_path_dims = [input_size * support_sizes[hop], self.num_neighbor, self.walk_dim]
                    neigh_vecs = tf.reshape(hidden_nodes[hop+1], neigh_node_dims)
                    path_vecs = tf.reshape(hidden_paths[hop], neigh_path_dims)
                    channel_amplifier = tf.nn.sigmoid(tf.einsum("ijk,kh->ijh", path_vecs, weight_path) + bias_path)
                    #attention_param = tf.nn.softmax(tf.einsum("ijk,k->ij", path_vecs, weight_attention), axis = 1)

                    
                    #neigh_mean = tf.reduce_mean(tf.einsum("ijk,ij->ijk",channel_amplifier * neigh_vecs,attention_param), axis = 1)
                    neigh_mean = tf.reduce_mean(channel_amplifier * neigh_vecs, axis = 1)
                    #neigh_mean = tf.reduce_mean(neigh_vecs, axis = 1)
                    from_neighs = tf.matmul(neigh_mean, weight_neigh)
                    from_self = tf.matmul(hidden_nodes[hop], weight_self)
                
                    if layer != 1:
                        final = tf.nn.relu(from_neighs + from_self + bias_aggregate)
                    else:
                        final = from_neighs + from_self + bias_aggregate
                    #final = tf.nn.l2_normalize(final, 1)
                    next_hidden.append(final)
            hidden_nodes = next_hidden
        #if compute_regularizer:
        #    return hidden_nodes[0], sum(l2_losses)
        #else:
        return hidden_nodes[0]

    def train(self):
        self.sess.run(tf.global_variables_initializer())

        def load_batch(q):
            while 1:
                batchkeys, batchlabels, batchnegs, walkkeys, walklabels, walknegs =\
                    self.Dataset.generate_batch(self.batch_size, self.window_size)
                q.put((batchkeys, batchlabels, batchnegs, walkkeys, walklabels, walknegs))
        
        q = queue.Queue(maxsize = 5)
        t = threading.Thread(target = load_batch, args = [q])
        t.daemon = True
        t.start()
        
        losses = []
        link_losses = []
        walk_losses = []

        for i in range(self.num_steps):
            keys, labels, negs, wkeys, wlabels, wnegs = q.get()

            _, batch_loss, batch_link_loss, batch_walk_loss = self.sess.run([
                self.opt_op, 
                self.loss, 
                self.link_loss, 
                self.walk_loss, 
            ], 
            feed_dict = {
                self.batch_keys: keys, 
                self.batch_labels: labels, 
                self.batch_negs: negs, 
                self.key_walks: wkeys, 
                self.label_walks: wlabels, 
                self.neg_walks: wnegs
            })
            losses.append(batch_loss)
            link_losses.append(batch_link_loss)
            walk_losses.append(batch_walk_loss)
            if i and i % 100 == 0:
                print("[%.2fs] After %d iters, loss, link loss, walk loss on training is %.4f, %.4f, %.4f."%(time.time()-self.start_time, i, np.mean(losses), np.mean(link_losses), np.mean(walk_losses)))
                losses = []
                link_losses = []
                walk_losses = []
            if i and i % 500 == 0:
                self.evaluate_model()
                self.save_embeddings(i, save_model= False)
    
    
    def get_full_embeddings(self):
        self.embedding_array = np.zeros((self.Dataset.num_nodes, self.embedding_dims))
        batch_size = 100
        for i in range(self.Dataset.num_nodes//batch_size + 1):
            if i != self.Dataset.num_nodes//batch_size:
                batchnode = np.arange(100*i, 100*i+100)
                batch_embeddings = self.sess.run([self.output], feed_dict = {
                    self.batch_input: batchnode, 
                    self.input_size: 100
                })
                self.embedding_array[100*i:100*i+100] = batch_embeddings[0]
            else:
                batchnode = np.arange(100*i, self.num_nodes)
                batch_embeddings = self.sess.run([self.output], feed_dict = {
                    self.batch_input: batchnode, 
                    self.input_size: self.num_nodes - 100*i
                })
                self.embedding_array[100*i:self.num_nodes] = batch_embeddings[0]
        return self.embedding_array

    
    def evaluate_model(self):
        # note, the evaluate_model is only used for tracking the training process and cannot be used 
        # for formal model evaluation

        self.get_full_embeddings()
        if self.purpose == "classification":
            macros = []
            micros = []
            for _ in range(10):
                validation_indice = random.sample(range(self.num_nodes), int(self.num_nodes * 0.7))
                train_indice = [x for x in range(self.num_nodes) if x not in validation_indice]


                train_feature = self.embedding_array[train_indice]
                train_label = self.Dataset.node2label[train_indice]
                validation_feature = self.embedding_array[validation_indice]
                validation_label = self.Dataset.node2label[validation_indice]


                clf = LogisticRegression(multi_class="auto", solver = "lbfgs", max_iter=500)
                # macros = cross_val_score(clf, self.embedding_array, self.Dataset.node2label, cv = 5, scoring = "f1_macro")
                # micros = cross_val_score(clf, self.embedding_array, self.Dataset.node2label, cv = 5, scoring = "f1_micro")
                clf.fit(train_feature, train_label)
                predict_label = clf.predict(validation_feature)
                macro_f1 = metrics.f1_score(validation_label, predict_label, average= "macro")
                micro_f1 = metrics.f1_score(validation_label, predict_label, average = "micro")
                macros.append(macro_f1)
                micros.append(micro_f1)
            print("Node classification macro f1: %.4f, std %.4f"%(np.mean(macros), np.std(macros)))
            print("Node classification micro f1: %.4f, std %.4f"%(np.mean(micros), np.std(micros)))   

        elif self.purpose == "link_prediction":
            truth_links = self.Dataset.pred_edge_list
            AUCs = []
            recalls = []
            for _ in range(5):
                negative_links = self.Dataset.sample_negative_links(len(truth_links))
                num = len(truth_links)
                ratings = []
                for s, d in truth_links:
                    ratings.append(self.embedding_array[s].dot(self.embedding_array[d]))
                for s, d in negative_links:
                    ratings.append(self.embedding_array[s].dot(self.embedding_array[d]))
                argsorted = np.argsort(ratings)
                recall = 0
                AUC = 0
                neg = 0
                for i in range(2*num):
                    if argsorted[i] < num:
                        AUC += neg/num
                        if i > num:
                            recall += 1
                    else:
                        neg += 1
                recall /= num
                AUC /= num
                AUCs.append(AUC)
                recalls.append(recall)
            print("Link prediction AUC: %.4f"%np.mean(AUCs))
            print("Link prediction recall@0.5: %.4f" % np.mean(recalls))
        elif self.purpose == "none":
            return


    def save_embeddings(self, step, save_model = True):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        np.save(self.save_path + "/" + str(step), arr = self.embedding_array)
        if save_model:
            self.saver.save(self.sess, self.save_path + "/model", global_step = step)
        print("Embedding saved for step #%d"%step)

    def restore(self, model_name):
        new_saver = tf.train.import_meta_graph(model_name + ".meta")
        new_saver.restore(self.sess, model_name)