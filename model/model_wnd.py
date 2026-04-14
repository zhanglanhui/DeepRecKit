# -*- encoding:utf-8 -*-
import six
import math
import tensorflow as tf
from tensorflow.python.estimator.canned import dnn
from tensorflow.python.estimator.canned import linear
from tensorflow.python.estimator.canned import optimizers
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.summary import summary
from tensorflow.python.training import training_util
from .estimator import ModelBase


WIDE_AND_DEEP_DNN_LEARNING_RATE = 0.001
WIDE_AND_DEEP_LINEAR_LEARNING_RATE = 0.005


def _get_dnn_partitioner(config):
    num_ps_replicas = config.num_ps_replicas if config else 0
    return partitioned_variables.min_max_variable_partitioner(
            max_partitions=num_ps_replicas)


def _get_layer_partitioner(config):
    num_ps_replicas = config.num_ps_replicas if config else 0
    input_layer_partitioner = partitioned_variables.min_max_variable_partitioner(
        max_partitions=num_ps_replicas,
        min_slice_size=64 << 20)
    return input_layer_partitioner


def _check_feature_columns(linear_feature_columns, dnn_feature_columns):
    if not linear_feature_columns and not dnn_feature_columns:
        raise ValueError(
            'Either linear_feature_columns or dnn_feature_columns must be defined.')
    feature_columns = (
            list(linear_feature_columns) + list(dnn_feature_columns))
    if not feature_columns:
        raise ValueError('Either linear_feature_columns or dnn_feature_columns '
                         'must be defined.')


def _linear_learning_rate(num_linear_feature_columns):
    default_learning_rate = 1. / math.sqrt(num_linear_feature_columns)
    return min(WIDE_AND_DEEP_LINEAR_LEARNING_RATE, default_learning_rate)


def _add_layer_summary(value, tag):
    summary.scalar('%s/fraction_of_zero_values' % tag, nn.zero_fraction(value))
    summary.histogram('%s/activation' % tag, value)


class WideAndDeepClassifier(ModelBase):
    def __init__(self,
                 wide_feature_columns=None,
                 wide_optimizer=None,
                 deep_feature_columns=None,
                 deep_optimizer=None,
                 output_dim=1,
                 deep_hidden_units=None,
                 deep_activation_fn=tf.nn.relu,
                 deep_dropout=None,
                 label_truncate=1.0):
        ModelBase.__init__(self, "WideAndDeepClassifier")
        self.linear_feature_columns = wide_feature_columns
        self.linear_optimizer = wide_optimizer
        self.dnn_feature_columns = deep_feature_columns
        self.dnn_optimizer = deep_optimizer
        self.dnn_hidden_units = deep_hidden_units
        self.dnn_activation_fn = deep_activation_fn
        self.dnn_dropout = deep_dropout
        self.output_dim = output_dim
        self.label_truncate = label_truncate
        if not self.dnn_hidden_units:
            raise ValueError(
                'dnn_hidden_units must be defined when dnn_feature_columns is '
                'specified.')
        _check_feature_columns(self.linear_feature_columns, self.dnn_feature_columns)
        if self.linear_optimizer is None:
            self.linear_optimizer = optimizers.get_optimizer_instance(
                "Ftrl", learning_rate=_linear_learning_rate(len(self.linear_feature_columns)))
        if self.dnn_optimizer is None:
            self.dnn_optimizer = optimizers.get_optimizer_instance(
                "Adagrad", learning_rate=WIDE_AND_DEEP_DNN_LEARNING_RATE)

    def get_optimizers(self):
        return [self.dnn_optimizer, self.linear_optimizer]

    def set_optimizers(self, optimizers):
        self.dnn_optimizer, self.linear_optimizer = optimizers

    def build_graph(self, features, targets, mode, config=None):
        input_layer_partitioner = _get_layer_partitioner(config)
        dnn_partitioner = _get_dnn_partitioner(config)
        # Deep Side
        dnn_parent_scope = 'dnn'
        if not self.dnn_feature_columns:
            dnn_logits = None
        else:
            with variable_scope.variable_scope(
                    dnn_parent_scope,
                    values=tuple(six.itervalues(features)),
                    partitioner=dnn_partitioner
            ):
                dnn_logit_fn = dnn._dnn_logit_fn_builder(
                    units=self.output_dim,
                    hidden_units=self.dnn_hidden_units,
                    feature_columns=self.dnn_feature_columns,
                    activation_fn=self.dnn_activation_fn,
                    dropout=self.dnn_dropout,
                    batch_norm=None,
                    input_layer_partitioner=input_layer_partitioner)
                dnn_logits = dnn_logit_fn(features=features, mode=mode)

        # Linear Side
        linear_parent_scope = 'linear'
        if not self.linear_feature_columns:
            linear_logits = None
        else:
            with variable_scope.variable_scope(
                    linear_parent_scope,
                    values=tuple(six.itervalues(features)),
                    partitioner=input_layer_partitioner
            ) as scope:
                logit_fn = linear._linear_logit_fn_builder(
                    units=self.output_dim,
                    feature_columns=self.linear_feature_columns)
                linear_logits = logit_fn(features=features)
                _add_layer_summary(linear_logits, scope.name)

        # Combine logits
        if dnn_logits is not None and linear_logits is not None:
            logits = dnn_logits + linear_logits
        elif dnn_logits is not None:
            logits = dnn_logits
        else:
            logits = linear_logits

        def _train_op_fn(loss):
            train_ops = []
            global_step = training_util.get_global_step()
            if dnn_logits is not None:
                train_ops.append(
                    self.dnn_optimizer.minimize(
                        loss, global_step=global_step,
                        var_list=ops.get_collection(
                            ops.GraphKeys.TRAINABLE_VARIABLES,
                            scope=dnn_parent_scope)))
            if linear_logits is not None:
                train_ops.append(
                    self.linear_optimizer.minimize(
                        loss, global_step=global_step,
                        var_list=ops.get_collection(
                            ops.GraphKeys.TRAINABLE_VARIABLES,
                            scope=linear_parent_scope)))
            train_op = control_flow_ops.group(*train_ops)
            with ops.control_dependencies([train_op]):
                with ops.colocate_with(global_step):
                    return state_ops.assign_add(global_step, 1)
        if self.output_dim <= 2:
            self.pre = tf.nn.sigmoid(logits)
        else:
            self.loss = tf.nn.softmax(logits)
        self.optimizer = None
        if targets is not None:
            if self.output_dim <= 2:
                self.loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=targets, logits=logits))
            else:
                self.loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=targets, logits=logits))
            self.train_op = _train_op_fn(self.loss)

    def get_pre(self):
        return self.pre

    def get_loss(self):
        return self.loss

    def get_optimizer(self):
        return self.optimizer

    def get_train_op(self):
        return self.train_op

    def get_metrics(self, targets, preds):
        targets_label = tf.where(tf.greater(targets, self.label_truncate), tf.ones_like(targets), tf.zeros_like(targets))
        return {
            'label/mean': tf.metrics.mean(targets),
            'prediction/mean': tf.metrics.mean(preds),
            "mean_absolute_error": tf.metrics.mean_absolute_error(targets, preds),
            "mean_squared_error": tf.metrics.mean_squared_error(targets, preds),
            'accuracy': tf.metrics.accuracy(labels=targets_label, predictions=tf.to_float(tf.greater_equal(preds, self.label_truncate))),
            'auc': tf.metrics.auc(targets_label, preds),
        }

    def get_eval_summary(self, targets, preds):
        summary = {}
        for key, value in self.get_metrics(targets, preds).items():
            summary[key] = value[1]
        return summary


class WideAndDeepRegressor(ModelBase):
    def __init__(self,
                 wide_feature_columns=None,
                 wide_optimizer=None,
                 deep_feature_columns=None,
                 deep_optimizer=None,
                 output_dim=1,
                 deep_hidden_units=None,
                 deep_activation_fn=tf.nn.relu,
                 deep_dropout=None):
        ModelBase.__init__(self, "WideAndDeepRegressor")
        self.linear_feature_columns = wide_feature_columns
        self.linear_optimizer = wide_optimizer
        self.dnn_feature_columns = deep_feature_columns
        self.dnn_optimizer = deep_optimizer
        self.dnn_hidden_units = deep_hidden_units
        self.dnn_activation_fn = deep_activation_fn
        self.dnn_dropout = deep_dropout
        self.output_dim = output_dim
        if not self.dnn_hidden_units:
            raise ValueError(
                'dnn_hidden_units must be defined when dnn_feature_columns is '
                'specified.')
        _check_feature_columns(self.linear_feature_columns, self.dnn_feature_columns)
        if self.linear_optimizer is None:
            self.linear_optimizer = optimizers.get_optimizer_instance(
                "Ftrl", learning_rate=_linear_learning_rate(len(self.linear_feature_columns)))
        if self.dnn_optimizer is None:
            self.dnn_optimizer = optimizers.get_optimizer_instance(
                "Adagrad", learning_rate=WIDE_AND_DEEP_DNN_LEARNING_RATE)

    def get_optimizers(self):
        return [self.dnn_optimizer, self.linear_optimizer]

    def set_optimizers(self, optimizers):
        self.dnn_optimizer, self.linear_optimizer = optimizers

    def build_graph(self, features, targets, mode, config=None):
        input_layer_partitioner = _get_layer_partitioner(config)
        dnn_partitioner = _get_dnn_partitioner(config)
        # Deep Side
        dnn_parent_scope = 'dnn'
        if not self.dnn_feature_columns:
            dnn_logits = None
        else:
            with variable_scope.variable_scope(
                    dnn_parent_scope,
                    values=tuple(six.itervalues(features)),
                    partitioner=dnn_partitioner
            ):
                dnn_logit_fn = dnn._dnn_logit_fn_builder(
                    units=self.output_dim,
                    hidden_units=self.dnn_hidden_units,
                    feature_columns=self.dnn_feature_columns,
                    activation_fn=self.dnn_activation_fn,
                    dropout=self.dnn_dropout,
                    batch_norm=None,
                    input_layer_partitioner=input_layer_partitioner)
                dnn_logits = dnn_logit_fn(features=features, mode=mode)

        # Linear Side
        linear_parent_scope = 'linear'
        if not self.linear_feature_columns:
            linear_logits = None
        else:
            with variable_scope.variable_scope(
                    linear_parent_scope,
                    values=tuple(six.itervalues(features)),
                    partitioner=input_layer_partitioner
            ) as scope:
                logit_fn = linear._linear_logit_fn_builder(
                    units=self.output_dim,
                    feature_columns=self.linear_feature_columns)
                linear_logits = logit_fn(features=features)
                _add_layer_summary(linear_logits, scope.name)

        # Combine logits
        if dnn_logits is not None and linear_logits is not None:
            logits = dnn_logits + linear_logits
        elif dnn_logits is not None:
            logits = dnn_logits
        else:
            logits = linear_logits

        def _train_op_fn(loss):
            train_ops = []
            global_step = training_util.get_global_step()
            if dnn_logits is not None:
                train_ops.append(
                    self.dnn_optimizer.minimize(
                        loss, global_step=global_step,
                        var_list=ops.get_collection(
                            ops.GraphKeys.TRAINABLE_VARIABLES,
                            scope=dnn_parent_scope)))
            if linear_logits is not None:
                train_ops.append(
                    self.linear_optimizer.minimize(
                        loss, global_step=global_step,
                        var_list=ops.get_collection(
                            ops.GraphKeys.TRAINABLE_VARIABLES,
                            scope=linear_parent_scope)))
            train_op = control_flow_ops.group(*train_ops)
            with ops.control_dependencies([train_op]):
                with ops.colocate_with(global_step):
                    return state_ops.assign_add(global_step, 1)
        self.pre = logits
        self.optimizer = None
        if targets is not None:
            self.loss = tf.reduce_sum(tf.square(self.pre - targets))
            self.train_op = _train_op_fn(self.loss)

    def get_pre(self):
        return self.pre

    def get_loss(self):
        return self.loss

    def get_optimizer(self):
        return self.optimizer

    def get_train_op(self):
        return self.train_op

    def get_metrics(self, targets, preds):
        return {
            'label/mean': tf.metrics.mean(targets),
            'prediction/mean': tf.metrics.mean(preds),
            "mean_absolute_error": tf.metrics.mean_absolute_error(targets, preds),
            "mean_squared_error": tf.metrics.mean_squared_error(targets, preds)
        }

    def get_eval_summary(self, targets, preds):
        summary = {}
        for key, value in self.get_metrics(targets, preds).items():
            summary[key] = value[1]
        return summary
