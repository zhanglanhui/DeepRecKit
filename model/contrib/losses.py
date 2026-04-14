#!/usr/bin/python

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import sys
import tensorflow as tf
import tensorflow.keras as K

# sys.path.append(".")
# sys.path.append("..")


class MaskedInfoNceLoss:
    def __init__(self):
        pass

    def _compute_mask(self, item_vec):
        # (batch, 1)
        item_id = item_vec[:, :1]

        # (batch, batch)
        item_diff = tf.abs(item_id - tf.transpose(item_id))
        same_item_mask = tf.cast(tf.less(item_diff, 1e-9), tf.float32) - tf.eye(
            tf.shape(item_vec)[0]
        )

        return same_item_mask

    def __call__(self, user_vec, item_vec, temp):
        same_item_mask = self._compute_mask(item_vec)

        # [batch, batch]
        outer_product = tf.matmul(user_vec, item_vec, transpose_b=True)

        masked_outer_product = outer_product * (1.0 - same_item_mask)

        # [batch, 1]
        nce_numerator = tf.expand_dims(
            tf.linalg.diag_part(masked_outer_product), axis=-1
        )

        # [batch, 1]
        nce_denominator = tf.reduce_sum(
            tf.exp(masked_outer_product / temp), axis=1, keepdims=True
        )

        # loss = -E[(log - log_sum)]
        # [batch,]
        nce_loss = -(nce_numerator / temp - tf.math.log(nce_denominator))

        return nce_loss, outer_product, nce_numerator, same_item_mask

class InfoNceLoss:
    def __init__(self):
        pass

    def __call__(self, user_vec, item_vec, temp):
        # [batch, batch]
        outer_product = tf.matmul(user_vec, item_vec, transpose_b=True)

        # [batch, 1]
        nce_numerator = tf.expand_dims(tf.linalg.diag_part(outer_product), axis=-1)

        # [batch, 1]
        nce_denominator = tf.reduce_sum(
            tf.exp(outer_product / temp), axis=1, keepdims=True
        )

        # loss = -E[(log - log_sum)]
        # [batch,]
        nce_loss = -(nce_numerator / temp - tf.math.log(nce_denominator))

        return nce_loss, outer_product, nce_numerator

def compute_info_nce_loss(user_vec, item_vec, temp):
    return InfoNceLoss()(user_vec, item_vec, temp)

def compute_masked_info_nce_loss(user_vec, item_vec, temp):
    return MaskedInfoNceLoss()(user_vec, item_vec, temp)

def compute_weighted_log_loss(pred, label, weight=None):
    loss = tf.expand_dims(K.losses.binary_crossentropy(label, pred), axis=-1)

    if weight is not None:
        loss = loss * weight

    return loss
