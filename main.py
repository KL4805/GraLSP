import os
import numpy as np
from models import *
import tensorflow as tf
from dataset import *
flags = tf.app.flags

flags.DEFINE_string("dataset_name", "cora", "The dataset to use, corresponds to a folder under data/")
flags.DEFINE_integer("path_length", 10, "The length of random_walks")
flags.DEFINE_integer("num_paths", 100, "The number of paths to use per node")
flags.DEFINE_integer("window_size", 6, "The window size to sample neighborhood")
flags.DEFINE_integer("batch_size", 100, "batch size")
flags.DEFINE_integer("neg_size", 8, "negative sampling size")
flags.DEFINE_float("learning_rate", 0.002, "learning rate")
flags.DEFINE_string("optimizer", "Adam", "The optimizer to use")
flags.DEFINE_integer("embedding_dims", 32, "The size of each embedding")
flags.DEFINE_integer("num_steps", 5001, "Steps to train")
flags.DEFINE_integer("num_skips", 5, "how many samples to draw from a single walk")
flags.DEFINE_integer("num_neighbor", 20, "How many neighbors to sample, for graphsage")
flags.DEFINE_integer("hidden_dim", 100, "The size of hidden dimension, for graphsage")
flags.DEFINE_integer("walk_dim", 30, "The size of embeddings for anonym. walks.")
flags.DEFINE_integer("anonym_walk_len", 8, "The length of each anonymous walk, 4 or 5")
flags.DEFINE_float("walk_loss_lambda", 0.1, "Weight of loss focusing on anonym walk similarity")
flags.DEFINE_string("purpose", "classification", "Tasks for evaluation, classification or link_prediction")
flags.DEFINE_float("linkpred_ratio", 0.1, "The ratio of edges being removed for link prediction")
flags.DEFINE_float("p", 0.25, "return parameter for node2vec walk")
flags.DEFINE_float("q", 1, "out parameter for node2vec walk")
flags.DEFINE_integer("inductive", 0, "whether to do inductive inference")
flags.DEFINE_integer("inductive_model_epoch", None, "the epoch of the saved model")
flags.DEFINE_string("inductive_model_name", None, "the path towards the loaded model")

FLAGS = flags.FLAGS


def main(_):

    print("Dataset: %s"%FLAGS.dataset_name)
    print("hidden dimension: %d"%FLAGS.hidden_dim)
    print("Lambda for walk loss: %f"%FLAGS.walk_loss_lambda)
    print("Anonym walk length: %d"%FLAGS.anonym_walk_len)
    sess = tf.Session()




    save_path = "embeddings/GraLSP"
    model = GraLSP(sess, FLAGS.dataset_name, FLAGS.purpose, FLAGS.linkpred_ratio, FLAGS.path_length,
        FLAGS.num_paths, FLAGS.window_size, FLAGS.batch_size, FLAGS.neg_size, FLAGS.learning_rate,
        FLAGS.optimizer, FLAGS.embedding_dims, save_path, FLAGS.num_steps, FLAGS.num_skips, FLAGS.hidden_dim,
        FLAGS.num_neighbor, FLAGS.anonym_walk_len, FLAGS.walk_loss_lambda,  FLAGS.walk_dim,
        FLAGS.p, FLAGS.q)
    if not FLAGS.inductive:
        model.train()
    else:
        model.restore(FLAGS.inductive_model_name)
        model.get_full_embeddings()
        model.save_embeddings(FLAGS.inductive_model_epoch, save_model=False)


if __name__ == "__main__":
    tf.app.run()
