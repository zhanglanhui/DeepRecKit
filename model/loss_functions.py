# -*- encoding:utf-8 -*-

import tensorflow as tf
from functools import partial

class LossFunc:
    @staticmethod
    def _binary_focal_loss_from_logits(labels, logits, gamma=2.0, pos_weight=0.25,
                                       label_smoothing=None):
        """Compute focal loss from logits using a numerically stable formula.
        Parameters
        ----------
        labels : tensor-like
            Tensor of 0's and 1's: binary class labels.
        logits : tf.Tensor
            Logits for the positive class.
        gamma : float
            Focusing parameter.
        pos_weight : float or None
            If not None, losses for the positive class will be scaled by this
            weight.
        label_smoothing : float or None
            Float in [0, 1]. When 0, no smoothing occurs. When positive, the binary
            ground truth labels `y_true` are squeezed toward 0.5, with larger values
            of `label_smoothing` leading to label values closer to 0.5.
        Returns
        -------
        tf.Tensor
            The loss for each example.
        """
        labels = LossFunc._process_labels(labels=labels, label_smoothing=label_smoothing,
                                          dtype=logits.dtype)

        # Compute probabilities for the positive class
        p = tf.math.sigmoid(logits)

        # Without label smoothing we can use TensorFlow's built-in per-example cross
        # entropy loss functions and multiply the result by the modulating factor.
        # Otherwise, we compute the focal loss ourselves using a numerically stable
        # formula below
        if label_smoothing is None:
            # The labels and logits tensors' shapes need to be the same for the
            # built-in cross-entropy functions. Since we want to allow broadcasting,
            # we do some checks on the shapes and possibly broadcast explicitly
            # Note: tensor.shape returns a tf.TensorShape, whereas tf.shape(tensor)
            # returns an int tf.Tensor; this is why both are used below
            labels_shape = labels.shape
            logits_shape = logits.shape
            if not labels_shape.is_fully_defined() or labels_shape != logits_shape:
                labels_shape = tf.shape(labels)
                logits_shape = tf.shape(logits)
                shape = tf.broadcast_dynamic_shape(labels_shape, logits_shape)
                labels = tf.broadcast_to(labels, shape)
                logits = tf.broadcast_to(logits, shape)
            if pos_weight is None:
                loss_func = tf.nn.sigmoid_cross_entropy_with_logits
            else:
                loss_func = partial(tf.nn.weighted_cross_entropy_with_logits,
                                    pos_weight=pos_weight)
            loss = loss_func(labels=labels, logits=logits)
            modulation_pos = (1 - p) ** gamma
            modulation_neg = p ** gamma
            mask = tf.dtypes.cast(labels, dtype=tf.bool)
            modulation = tf.where(mask, modulation_pos, modulation_neg)
            return modulation * loss

        # Terms for the positive and negative class components of the loss
        pos_term = labels * ((1 - p) ** gamma)
        neg_term = (1 - labels) * (p ** gamma)

        # Term involving the log and ReLU
        log_weight = pos_term
        if pos_weight is not None:
            log_weight *= pos_weight
        log_weight += neg_term
        log_term = tf.math.log1p(tf.math.exp(-tf.math.abs(logits)))
        log_term += tf.nn.relu(-logits)
        log_term *= log_weight

        # Combine all the terms into the loss
        loss = neg_term * logits + log_term
        return loss

    @staticmethod
    def _process_labels(labels, label_smoothing, dtype):
        """Pre-process a binary label tensor, maybe applying smoothing.
        Parameters
        ----------
        labels : tensor-like
            Tensor of 0's and 1's.
        label_smoothing : float or None
            Float in [0, 1]. When 0, no smoothing occurs. When positive, the binary
            ground truth labels `y_true` are squeezed toward 0.5, with larger values
            of `label_smoothing` leading to label values closer to 0.5.
        dtype : tf.dtypes.DType
            Desired type of the elements of `labels`.
        Returns
        -------
        tf.Tensor
            The processed labels.
        """
        labels = tf.dtypes.cast(labels, dtype=dtype)
        if label_smoothing is not None:
            labels = (1 - label_smoothing) * labels + label_smoothing * 0.5
        return labels
