from tensorflow.python.ops import math_ops
import tensorflow as tf
from tensorflow.python.ops import math_ops

# safe_embedding_lookup_sparse(
#                         self.embeddings[k], sparse_ids, sparse_weights=sparse_weights, combiner="sum")
def safe_embedding_lookup_sparse(embeddings, sparse_ids, sparse_weights, combiner, embedding_fun=math_ops.abs):
    sparse_embeds = tf.nn.embedding_lookup(embeddings, sparse_ids)
    if embedding_fun is not None:
        sparse_embeds = embedding_fun(sparse_embeds)
    if combiner == "sum":
        output = tf.reduce_sum(tf.multiply(sparse_embeds, sparse_weights), axis=1)
    elif combiner == "mean":
        output = tf.reduce_mean(tf.multiply(sparse_embeds, sparse_weights), axis=1)
    else:
        raise ValueError("Unknown combiner: {}".format(combiner))

    return output
