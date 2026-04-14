# -*- encoding:utf-8 -*-

import tensorflow as tf


def sequence_input_layer(features, items_list):
    embeddings = []
    lengths = []
    for items in items_list:
        item_embedding = tf.feature_column.input_layer(features, [items])
        item_length = tf.reduce_sum(
            tf.cast(tf.not_equal(features[items.name], 0), tf.int32),
            axis=1
        )
        embeddings.append(item_embedding)
        lengths.append(item_length)

    return embeddings, lengths
