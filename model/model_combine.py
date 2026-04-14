# -*- encoding:utf-8 -*-

import copy
import tensorflow as tf
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import training_util
from tensorflow.python.framework import ops
from tensorflow.python.ops import state_ops
from .estimator import ModelBase


class DeepCombineNetwork(ModelBase):
    WIDE_SCOPE = "wide"
    DEEP_CROSS_SCOPE = "deep_cross"
    DEFAULT_HIDDEN_UNITS = [512, 68]
    DEFAULT_WIDE_LR = 0.0001
    DEFAULT_DEEP_LR = 0.0001
    DEFAULT_DEEP_DROPOUT = 0.15
    DEFAULT_CROSS_DROPOUT = 0.10
    DEFAULT_CROSS_LAYERS = 3

    def __init__(self,
                 wide_feature_columns,
                 deep_feature_columns,
                 cross_feature_columns,
                 output_dim=1,
                 use_wide=True,
                 use_cross=True,
                 use_deep=True,
                 dcn_parallel=True,
                 deep_use_bn=False,
                 cross_use_bn=False,
                 wide_learning_rate=DEFAULT_WIDE_LR,
                 wide_optimizer=None,
                 deep_learning_rate=DEFAULT_DEEP_LR,
                 deep_optimizer=None,
                 deep_dropout=DEFAULT_DEEP_DROPOUT,
                 deep_activate_fn=tf.nn.relu,
                 cross_dropout=DEFAULT_CROSS_DROPOUT,
                 deep_hidden_units=None,
                 cross_layers=DEFAULT_CROSS_LAYERS):
        ModelBase.__init__(self, "DeepCombineNetwork")
        if (not use_cross) and (not use_deep) and (not use_wide):
            raise Exception("One Unit Needed Must")
        self.wide_feature_columns = wide_feature_columns
        self.deep_feature_columns = deep_feature_columns
        self.cross_feature_columns = cross_feature_columns
        self.use_wide = use_wide
        self.use_cross = use_cross
        self.use_deep = use_deep
        self.deep_use_bn = deep_use_bn
        self.cross_use_bn = cross_use_bn
        self.wide_learning_rate = wide_learning_rate
        self.deep_learning_rate = deep_learning_rate
        self.output_dim = output_dim
        self.dcn_parallel = dcn_parallel
        if deep_optimizer is None:
            self.deep_optimizer = tf.train.AdagradOptimizer(learning_rate=self.deep_learning_rate)
        else:
            self.deep_optimizer = deep_optimizer
        if wide_optimizer is None:
            self.wide_optimizer = tf.train.FtrlOptimizer(learning_rate=self.wide_learning_rate)
        else:
            self.wide_optimizer = wide_optimizer
        self.deep_dropout = deep_dropout
        self.cross_dropout = cross_dropout
        self.deep_activate_fn = deep_activate_fn
        if deep_hidden_units is None:
            self.deep_hidden_units = self.DEFAULT_HIDDEN_UNITS
        else:
            self.deep_hidden_units = deep_hidden_units
        self.cross_layers = cross_layers
        self.pre = None
        self.loss = None

    def build_graph(self, features, targets, mode=None, config=None):
        is_training = True
        if mode == tf.estimator.ModeKeys.PREDICT or mode == tf.estimator.ModeKeys.EVAL:
            self.deep_dropout = 0.0
            self.cross_dropout = 0.0
            is_training = False
        num_ps_replicas = config.num_ps_replicas if config else 0
        partitioner = partitioned_variables.min_max_variable_partitioner(
            max_partitions=num_ps_replicas,
            min_slice_size=64 << 20)
        output_tensor = self.get_output_tensor(self.wide_feature_columns, self.deep_feature_columns, self.cross_feature_columns, features, partitioner, is_training)
        if self.output_dim <= 2:
            self.pre = tf.nn.sigmoid(output_tensor)
        else:
            self.pre = tf.nn.softmax(output_tensor)
        self.loss = None
        if targets is not None:
            if self.output_dim <= 2:
                self.loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=targets, logits=output_tensor))
            else:
                self.loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=targets, logits=output_tensor))
            self.train_op = self._train_op_fn(self.loss)

    def get_output_tensor(self, wide_feature_columns, deep_feature_columns, cross_feature_columns, features, partitioner, is_training):
        output_tensor = 0
        if self.use_wide:
            with tf.variable_scope(self.WIDE_SCOPE, reuse=tf.AUTO_REUSE, partitioner=partitioner):
                wide_output = tf.feature_column.linear_model(features, wide_feature_columns, units=self.output_dim)
            output_tensor += wide_output
        with tf.variable_scope(self.DEEP_CROSS_SCOPE, reuse=tf.AUTO_REUSE):
            if self.use_deep and self.use_cross and not self.dcn_parallel:
                cross_input = tf.feature_column.input_layer(features, cross_feature_columns)
                if self.cross_use_bn:
                    cross_input = tf.layers.batch_normalization(cross_input, training=is_training)
                cross_output = self.get_cross_layer(cross_input, self.cross_layers)
                deep_input = tf.feature_column.input_layer(features, deep_feature_columns)
                if self.deep_use_bn:
                    deep_input = tf.layers.batch_normalization(deep_input, training=is_training)
                units = list(copy.copy(self.deep_hidden_units))
                deep_output_tmp = self.build_deep_layers(tf.concat([cross_output, deep_input], 1), units, self.deep_activate_fn, self.deep_dropout)
                output_tensor += self.build_deep_layers(deep_output_tmp, [self.output_dim], None)
            else:
                if self.use_deep:
                    deep_input = tf.feature_column.input_layer(features,  deep_feature_columns)
                    if self.deep_use_bn:
                        deep_input = tf.layers.batch_normalization(deep_input, training=is_training)
                    units = list(copy.copy(self.deep_hidden_units))
                    deep_output_tmp = self.build_deep_layers(deep_input, units, self.deep_activate_fn, self.deep_dropout)
                    deep_output = self.build_deep_layers(deep_output_tmp, [self.output_dim], None)
                    output_tensor += deep_output
                if self.use_cross:
                    cross_input = tf.feature_column.input_layer(features, cross_feature_columns)
                    if self.cross_use_bn:
                        cross_input = tf.layers.batch_normalization(cross_input, training=is_training)
                    cross_output = self.build_deep_layers(self.get_cross_layer(cross_input, self.cross_layers), [self.output_dim], None)
                    output_tensor += cross_output
        return output_tensor

    def _train_op_fn(self, loss):
        train_ops = []
        global_step = training_util.get_global_step()
        if self.use_wide:
            train_ops.append(self.wide_optimizer.minimize(
                loss, global_step=global_step, var_list=ops.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES, scope=self.WIDE_SCOPE)))
        if self.use_cross or self.use_deep:
            train_ops.append(self.deep_optimizer.minimize(
                loss, global_step=global_step, var_list=ops.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES, scope=self.DEEP_CROSS_SCOPE)))
        train_op = control_flow_ops.group(*train_ops)
        with ops.control_dependencies([train_op]):
            with ops.colocate_with(global_step):
                return state_ops.assign_add(global_step, 1)

    def get_cross_layer(self, input_tensor, cross_layers):
        dims = input_tensor.shape[1]
        origin_tensor = input_tensor
        output_tensor = origin_tensor
        for i in range(cross_layers):
            w = tf.get_variable(name=self.DEEP_CROSS_SCOPE + "_cross_w_%d" % i, shape=[dims], dtype=tf.float32,
                                initializer=tf.random_normal_initializer(mean=0.0, stddev=1.0, seed=None, dtype=tf.float32))
            b = tf.get_variable(name=self.DEEP_CROSS_SCOPE + "_cross_b_%d" % i, shape=[dims], dtype=tf.float32,
                                initializer=tf.random_normal_initializer(mean=0.0, stddev=1.0, seed=None, dtype=tf.float32))
            trans = tf.reshape(output_tensor, [-1, 1, dims])
            dot = tf.tensordot(trans, w, axes=1)
            output_tensor = origin_tensor * dot + b + output_tensor
            if 1.0 > self.cross_dropout > 0.0:
                output_tensor = tf.nn.dropout(output_tensor, keep_prob=1-self.cross_dropout)
        return output_tensor

    def get_pre(self,):
        return self.pre

    def get_loss(self):
        return self.loss

    def get_optimizer(self):
        return self.optimizer

    def get_metrics(self, targets, preds):
        return {
            'label/mean': tf.metrics.mean(targets),
            'prediction/mean': tf.metrics.mean(preds),
            "mean_absolute_error": tf.metrics.mean_absolute_error(targets, preds),
            "mean_squared_error": tf.metrics.mean_squared_error(targets, preds),
            'accuracy': tf.metrics.accuracy(labels=targets, predictions=tf.to_float(tf.greater_equal(preds, 0.5))),
            'auc': tf.metrics.auc(targets, preds),
        }

    def get_eval_summary(self, targets, preds):
        summary = {}
        for key, value in self.get_metrics(targets, preds).items():
            summary[key] = value[1]
        return summary
