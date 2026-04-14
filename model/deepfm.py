# -*- encoding:utf-8 -*-
# @Author: gavinzhang
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
import six
import collections
import tensorflow as tf
from tensorflow.python.estimator.canned import dnn
from tensorflow.python.estimator.canned import linear
from tensorflow.python.estimator.canned import optimizers
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import variable_scope
from tensorflow.python.training import sync_replicas_optimizer
from tensorflow.python.training import training_util
from tensorflow.python.summary import summary
from tensorflow.python.feature_column import feature_column
from .estimator import ModelBase
from model.loss_functions import LossFunc
from model.nn_layers import NNLayers

class DeepFactorMachineClassifier(ModelBase):
    """An estimator for TensorFlow DeepFM classification models.
    paper: "DeepFM: A Factorization-Machine based Neural Network for CTR Prediction"
    """
    WIDE_SCOPE = "sparse"
    DEEP_SCOPE = "dense"
    COMBINE_SCOPE = "combine"
    EXPERT_SCOPE = "expert"
    TASK_SCOPE = "task"
    DEFAULT_HIDDEN_UNITS = [128, 64]
    DEFAULT_WIDE_LR = 0.02
    DEFAULT_DEEP_LR = 0.01
    DEFAULT_DEEP_DROPOUT = 0.0
    DEFAULT_CROSS_DROPOUT = 0.0
    DEFAULT_CROSS_LAYERS = 2
    EMBEDDING_SIZE = 128

    def __init__(self,
                 deep_feature_columns,
                 wide_feature_columns,
                 weight_input_features,
                 task_names,
                 expert_units=None,
                 task_units=None,
                 concat_tower_units=None,
                 wide_optimizer=None,
                 deep_optimizer=None,
                 use_focal_loss=False,
                 use_cross=False,
                 cross_feature_columns=None,
                 cross_layers=DEFAULT_CROSS_LAYERS,
                 cross_l2=0.05,
                 cross_matrix=False,
                 expert_dropout=DEFAULT_DEEP_DROPOUT,
                 expert_activate_fn=None,
                 update_all_task=True,
                 block_gradient=True):
        super(DeepFactorMachineClassifier, self).__init__(model_name="DeepFM")

        if task_units is None:
            task_units = [128, 64]
        if concat_tower_units is None:
            concat_tower_units = [128, 64]
        if expert_units is None:
            expert_units = [128, 64]
        # print("deep_feature_columns", deep_feature_columns)
        self.deep_feature_columns = deep_feature_columns
        self.wide_feature_columns = wide_feature_columns
        self.cross_feature_columns = cross_feature_columns
        self.weight_input_features = weight_input_features
        self.task_names = task_names
        self.expert_units = expert_units
        self.task_units = task_units
        self.concat_tower_units = concat_tower_units
        self.use_focal_loss = use_focal_loss
        self.wide_learning_rate = DeepFactorMachineClassifier.DEFAULT_WIDE_LR
        self.deep_learning_rate = DeepFactorMachineClassifier.DEFAULT_DEEP_LR

        self.cross_dropout = DeepFactorMachineClassifier.DEFAULT_CROSS_DROPOUT
        self.use_cross = use_cross
        self.cross_matrix = cross_matrix
        self.cross_use_bn = False
        self.deep_use_bn = False

        self.expert_dropout = expert_dropout
        self.cross_layers = cross_layers
        self.cross_l2 = cross_l2
        self.print_ops = []
        if expert_activate_fn:
            self.expert_activate_fn = expert_activate_fn
        else:
            self.expert_activate_fn = tf.nn.relu
        self.pre = {}

        linear_feature_columns = wide_feature_columns or []
        dnn_feature_columns = deep_feature_columns or []
        self._feature_columns = (
                list(linear_feature_columns) + list(dnn_feature_columns))
        if not self._feature_columns:
            raise ValueError('Either linear_feature_columns or dnn_feature_columns '
                             'must be defined.')
        if deep_optimizer is None:
            self.deep_optimizer = tf.train.AdagradOptimizer(
                learning_rate=self.deep_learning_rate)
        else:
            self.deep_optimizer = deep_optimizer
        if wide_optimizer is None:
            self.wide_optimizer = tf.train.FtrlOptimizer(learning_rate=self.wide_learning_rate)
        else:
            self.wide_optimizer = wide_optimizer

    def get_deep_layer(self, features, deep_feature_columns, cross_feature_columns, is_training):
        deep_input = tf.feature_column.input_layer(features, deep_feature_columns)
        if self.use_cross:
            cross_input = tf.feature_column.input_layer(features, cross_feature_columns)
            if self.cross_use_bn:
                cross_input = tf.layers.batch_normalization(cross_input, training=is_training)
            if self.cross_matrix:
                cross_output = NNLayers.get_cross_matrix_layer(input_tensor=cross_input, cross_layers=self.cross_layers,
                                                               cross_dropout=self.cross_dropout, name="cross")
            else:
                cross_output = NNLayers.get_cross_layer(input_tensor=cross_input, cross_layers=self.cross_layers,
                                                        cross_dropout=self.cross_dropout, name="cross")
            deep_input = tf.concat([deep_input, cross_output], axis=-1)

        if self.deep_use_bn:
            deep_input = tf.layers.batch_normalization(deep_input, training=is_training)
        deep_output = NNLayers.build_deep_layers(deep_input, self.expert_units, self.expert_activate_fn,
                                                 self.expert_dropout, "expert")
        return deep_output

    def build_graph(self, features, targets, mode=None, config=None):
        self.mode = mode
        is_training = False
        if mode == tf.estimator.ModeKeys.PREDICT or mode == tf.estimator.ModeKeys.EVAL:
            self.deep_dropout = 0.0
            self.cross_dropout = 0.0
            is_training = True
        if not mode:
            self.mode = tf.estimator.ModeKeys.TRAIN

        targets1 = None
        if self.mode == tf.estimator.ModeKeys.TRAIN or self.mode == tf.estimator.ModeKeys.EVAL:
            # print("targets.shape()", targets["label"].shape)
            targets1 = tf.reshape(targets["label"], (-1, 1))

        with tf.variable_scope(self.WIDE_SCOPE, reuse=tf.AUTO_REUSE):
            linear_logits = tf.feature_column.linear_model(features, self.wide_feature_columns)
        with tf.variable_scope(self.DEEP_SCOPE, reuse=tf.AUTO_REUSE):
            deep_output = self.get_deep_layer(features, self.deep_feature_columns, self.cross_feature_columns,
                                              is_training)
            deep_logits = NNLayers.build_deep_layers(deep_output, [1], None, 0.0, "task_out")

        # interaction_logits = _build_interaction_model(features, input_feature_columns, interaction_type)

        logits = linear_logits + deep_logits
        self.task1_sparse_pre = tf.nn.sigmoid(linear_logits)
        self.task1_dense_pre = tf.nn.sigmoid(deep_logits)
        self.pre["label"] = tf.nn.sigmoid(logits)
        if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
            self.loss = self.cal_loss(targets1, logits)
            self.train_op = self._train_op_fn(self.loss)
            return tf.estimator.EstimatorSpec(mode, loss=self.loss, train_op=self.train_op)
        elif mode == tf.estimator.ModeKeys.PREDICT:
            probabilities = tf.nn.sigmoid(logits)
            predictions = {'probabilities': probabilities}
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    def cal_loss(self, labels, logits):
        # cross_entropy = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits))
        # self.print_ops.append(tf.print("[DEBUG] ============= labels_to_append, pred_to_append: ", labels, logits))
        if self.use_focal_loss:
            loss = LossFunc._binary_focal_loss_from_logits(labels=labels, logits=logits)
        else:
            loss = LossFunc._sigmoid_cross_entropy_loss_from_logits(labels=labels, logits=logits)
            # self.print_ops.append(tf.print("[DEBUG] ============= loss before reduce_mean: ", loss))
        loss = tf.reduce_mean(loss)
        # self.print_ops.append(tf.print("[DEBUG] ============= loss after reduce_mean: ", loss1))
        return loss

    def get_eval_summary(self, targets, preds):
        summary = {}
        for key, value in self.get_metrics(targets, preds).items():
            summary[key] = value[1]
        return summary

    def get_pre(self, ):
        return self.pre

    def get_loss(self):
        return self.loss

    def get_metrics(self, targets, preds):
        task_pre = preds["label"]
        task_label = targets["label"]
        return {
            'task_label/mean': tf.metrics.mean(task_label),
            'task_auc': tf.metrics.auc(task_label, task_pre),
            'task_wide_auc': tf.metrics.auc(task_label, self.task1_sparse_pre),
            'task_deep_auc': tf.metrics.auc(task_label, self.task1_dense_pre),
        }

    def _train_op_fn(self, loss, ):
        train_ops = []
        global_step = training_util.get_global_step()
        train_ops.append(self.deep_optimizer.minimize(loss, global_step=global_step,
                                                      var_list=ops.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES,
                                                                                  scope=self.DEEP_SCOPE)))
        train_ops.append(self.wide_optimizer.minimize(loss, global_step=global_step,
                                                      var_list=ops.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES,
                                                                                  scope=self.WIDE_SCOPE)))
        train_op = control_flow_ops.group(*train_ops)
        print_op = control_flow_ops.group(*self.print_ops)
        with ops.control_dependencies([train_op, print_op]):
            with ops.colocate_with(global_step):
                return state_ops.assign_add(global_step, 1)
