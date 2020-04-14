#Basic input pattern shape: (Batch, Letter_Index);
#Embedded input pattern shape: (Batch, Letter_Index, distributed_Pattern)

import tensorflow as tf;
import numpy as np;
from tensorflow.contrib.rnn import RNNCell, BasicRNNCell, LSTMCell, GRUCell, MultiRNNCell, LSTMStateTuple, OutputProjectionWrapper;
from tensorflow.contrib.seq2seq import AttentionWrapper, BahdanauMonotonicAttention, BasicDecoder, dynamic_decode, Helper, TrainingHelper;
from Hyper_Parameters import sound_Parameters;

def Cosine_Similarity(x,y):
    """
    Compute the cosine similarity between same row of two tensors.
    Args:
        x: nd tensor (...xMxN).
        y: nd tensor (...xMxN). A tensor of the same shape as x
    Returns:        
        cosine_Similarity: A (n-1)D tensor representing the cosine similarity between the rows. Size is (...xM)
    """
    return tf.reduce_sum(x * y, axis=-1) / (tf.sqrt(tf.reduce_sum(tf.pow(x, 2), axis=-1)) * tf.sqrt(tf.reduce_sum(tf.pow(y, 2), axis=-1)));

def Cosine_Similarity2D(x, y):
    """
    Compute the cosine similarity between each row of two tensors.
    Args:
        x: 2d tensor (MxN). The number of second dimension should be same to y's second dimension.
        y: 2d tensor (LxN). The number of second dimension should be same to x's second dimension.
    Returns:        
        cosine_Similarity: A `Tensor` representing the cosine similarity between the rows. Size is (M x L)
    """
    tiled_X = tf.tile(tf.expand_dims(x, [1]), multiples = [1, tf.shape(y)[0], 1]);   #[M, L, N]
    tiled_Y = tf.tile(tf.expand_dims(y, [0]), multiples = [tf.shape(x)[0], 1, 1]);   #[M, L, N]
    cosine_Similarity = tf.reduce_sum(tiled_Y * tiled_X, axis = 2) / (tf.sqrt(tf.reduce_sum(tf.pow(tiled_Y, 2), axis = 2)) * tf.sqrt(tf.reduce_sum(tf.pow(tiled_X, 2), axis = 2)) + 1e-8)  #[M, L]
    cosine_Similarity = tf.identity(cosine_Similarity, name="cosine_Similarity");

    return cosine_Similarity;

def Batch_Cosine_Similarity2D(x, y):    
    """
    Compute the cosine similarity between each row of two tensors.
    Args:
        x: 3d tensor (BATCHxMxN). The number of first and third dimension should be same to y's first and third dimension.
        y: 3d tensor (BATCHxLxN). The number of first and third dimension should be same to x's first and third dimension.
    Returns:        
        cosine_Similarity: A `Tensor` representing the cosine similarity between the rows. Size is (BATCH x M x L)
    """
    tiled_X = tf.tile(tf.expand_dims(x, [2]), multiples = [1, 1, tf.shape(y)[1], 1]);   #[Batch, M, L, N]
    tiled_Y = tf.tile(tf.expand_dims(y, [1]), multiples = [1, tf.shape(x)[1], 1, 1]);   #[Batch, M, L, N]
    cosine_Similarity = tf.reduce_sum(tiled_Y * tiled_X, axis = 3) / (tf.sqrt(tf.reduce_sum(tf.pow(tiled_Y, 2), axis = 3)) * tf.sqrt(tf.reduce_sum(tf.pow(tiled_X, 2), axis = 3)) + 1e-8)  #[Batch, M, L]
    cosine_Similarity = tf.identity(cosine_Similarity, name="cosine_Similarity");

    return cosine_Similarity;
