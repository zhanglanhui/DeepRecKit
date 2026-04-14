# -*- encoding:utf-8 -*-

import copy
from tensorflow.python.estimator.canned import linear
# from tensorflow.python.estimator.canned.dnn import _DNNModel
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import training_util
from tensorflow.python.framework import ops
from tensorflow.python.ops import state_ops
from .estimator import ModelBase
from model.activation_dice import *
from model.nn_layers import *

class MTLNetwork(ModelBase):
    WIDE_SCOPE = "wide"
    DEEP_SCOPE = "deep"
    EXPERT_SCOPE = "expert"
    TASK_SCOPE = "task"
    DEFAULT_HIDDEN_UNITS = [128]
    DEFAULT_DEEP_LR = 0.003
    DEFAULT_DEEP_DROPOUT = 0.15
    DEFAULT_CROSS_DROPOUT = 0.0
    DEFAULT_CROSS_LAYERS = 2
    ALL_MODEL = ['shared_bottom', 'one_gate', 'multi_gate']

    def __init__(self,
                 wide_feature_columns,
                 deep_feature_columns,
                 target_weight_column,
                 use_cross=False,
                 cross_feature_columns=None,
                 cross_dropout=DEFAULT_CROSS_DROPOUT,
                 cross_layers=DEFAULT_CROSS_LAYERS,
                 cross_l2=0.05,
                 cross_matrix=True,
                 task_num=2,
                 expert_num=3,
                 expert_hidden_units=DEFAULT_HIDDEN_UNITS,
                 expert_dropout=DEFAULT_DEEP_DROPOUT,
                 expert_activate_fn=None,
                 task_hidden_units=DEFAULT_HIDDEN_UNITS,
                 task_dropout=DEFAULT_DEEP_DROPOUT,
                 task_activate_fn=None,
                 learning_rate=DEFAULT_DEEP_LR,
                 wide_optimizer=None,
                 deep_optimizer=None,
                 update_all_task=True,
                 block_gradient=True,
                 model='shared_bottom',
                 ):
        ModelBase.__init__(self, "MTLNetwork")
        if deep_feature_columns is None:
            raise Exception("Please Input Deep Feature Columns")
        self.wide_feature_columns = wide_feature_columns
        self.deep_feature_columns = deep_feature_columns
        self.target_weight_column = target_weight_column
        self.task_num = task_num
        self.expert_num = expert_num
        self.expert_hidden_units = expert_hidden_units
        self.task_hidden_units = task_hidden_units
        self.learning_rate = learning_rate
        self.expert_dropout = expert_dropout
        self.use_cross = use_cross
        self.cross_feature_columns = cross_feature_columns
        self.cross_dropout = cross_dropout
        self.cross_layers = cross_layers
        self.cross_l2 = cross_l2
        self.cross_matrix = cross_matrix
        if expert_activate_fn:
            self.expert_activate_fn = expert_activate_fn
        else:
            self.expert_activate_fn = tf.nn.relu
        self.task_dropout = task_dropout
        if task_activate_fn:
            self.task_activate_fn = task_activate_fn
        else:
            self.task_activate_fn = tf.nn.relu
        if deep_optimizer is None:
            self.deep_optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate)
        else:
            self.deep_optimizer = deep_optimizer
        if wide_optimizer is None:
            self.wide_optimizer = tf.train.FtrlOptimizer(learning_rate=self.learning_rate)
        else:
            self.wide_optimizer = wide_optimizer
        self.update_all_task = update_all_task
        self.block_gradient = block_gradient
        self.pre = []
        self.wide_pre = []
        self.deep_pre = []
        self.pres = None
        self.wide_pres = None
        self.deep_pres = None
        if model not in MTLNetwork.ALL_MODEL:
            raise Exception("Unknown model")
        self.model = model
        self.loss_detail = []
        self.wide_loss_detail = []
        self.deep_loss_detail = []
        self.wide_loss = None
        self.deep_loss = None

    def build_graph(self, features, targets, mode=None, config=None):
        num_ps_replicas = config.num_ps_replicas if config else 0
        partitioner = partitioned_variables.min_max_variable_partitioner(
            max_partitions=num_ps_replicas,
            min_slice_size=64 << 20)
        self.wide_pres = self.cal_wide_output(features, partitioner)
        self.deep_pres = self.cal_deep_output(features, partitioner)
        self.pres = []
        for i in range(self.task_num):
            wide_pre = tf.sigmoid(self.wide_pres[i])
            deep_pre = tf.sigmoid(self.deep_pres[i])
            pre = tf.sigmoid(self.wide_pres[i] + self.deep_pres[i])
            self.pres.append(self.wide_pres[i] + self.deep_pres[i])
            self.pre.append(pre[:, 0:1])
            self.wide_pre.append(wide_pre)
            self.deep_pre.append(deep_pre)
        if targets is not None:
            self.loss, self.loss_detail = self.cal_loss(features, self.pres, targets)
            self.wide_loss, self.wide_loss_detail = self.cal_loss(features, self.wide_pres, targets)
            self.deep_loss, self.deep_loss_detail = self.cal_loss(features, self.deep_pres, targets)
            self.train_op = self._train_op_fn(self.loss, self.loss_detail)

    def cal_loss(self, features, predicts, targets):
        loss_detail = []
        loss = 0
        weight = tf.feature_column.input_layer(features, self.target_weight_column)
        for i in range(self.task_num):
            if self.update_all_task:
                loss_tmp = tf.reduce_sum(
                    tf.nn.sigmoid_cross_entropy_with_logits(logits=predicts[i], labels=targets[:, i:i + 1]))
            else:
                loss_tmp = tf.reduce_sum(
                    tf.nn.sigmoid_cross_entropy_with_logits(logits=predicts[i], labels=targets[:, i:i + 1]) * weight[:,
                                                                                                              i:i + 1])
            loss_detail.append(loss_tmp)
            loss += loss_tmp
        return loss, loss_detail

    def _train_op_fn(self, loss, loss_detail):
        train_ops = []
        global_step = training_util.get_global_step()
        if self.block_gradient:
            for i in range(self.task_num):
                train_ops.append(self.deep_optimizer.minimize(loss_detail[i], global_step=global_step,
                                                              var_list=ops.get_collection(
                                                                  ops.GraphKeys.TRAINABLE_VARIABLES,
                                                                  scope="%s_%s_%d" % (
                                                                      self.DEEP_SCOPE, self.TASK_SCOPE, i))))
            for i in range(self.task_num):
                train_ops.append(self.wide_optimizer.minimize(loss_detail[i], global_step=global_step,
                                                              var_list=ops.get_collection(
                                                                  ops.GraphKeys.TRAINABLE_VARIABLES,
                                                                  scope="%s_%s_%d" % (
                                                                      self.WIDE_SCOPE, self.TASK_SCOPE, i))))
            if self.model != "shared_bottom":
                train_ops.append(self.deep_optimizer.minimize(loss, global_step=global_step,
                                                              var_list=ops.get_collection(
                                                                  ops.GraphKeys.TRAINABLE_VARIABLES, scope="%s_%s" % (
                                                                      self.DEEP_SCOPE, self.EXPERT_SCOPE))))
        else:
            train_ops.append(self.deep_optimizer.minimize(loss, global_step=global_step,
                                                          var_list=ops.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES,
                                                                                      scope=".*%s.*" % self.DEEP_SCOPE)))
            train_ops.append(self.wide_optimizer.minimize(loss, global_step=global_step,
                                                          var_list=ops.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES,
                                                                                      scope=".*%s.*" % self.WIDE_SCOPE)))
        train_op = control_flow_ops.group(*train_ops)
        with ops.control_dependencies([train_op]):
            with ops.colocate_with(global_step):
                return state_ops.assign_add(global_step, 1)

    # wide_task_{n}
    def cal_wide_output(self, features, partitioner):
        outputs = []
        if self.wide_feature_columns:
            for i in range(self.task_num):
                with tf.variable_scope(MTLNetwork.WIDE_SCOPE + "_" + MTLNetwork.TASK_SCOPE + "_%d" % i,
                                       reuse=tf.AUTO_REUSE, partitioner=partitioner):
                    output = tf.feature_column.linear_model(features, self.wide_feature_columns, units=1)
                outputs.append(output)
        return outputs

    # deep_expert deep_task_{n}
    def cal_deep_output(self, features, partitioner):
        if self.model == 'shared_bottom':
            task_output = []
            with tf.variable_scope(MTLNetwork.DEEP_SCOPE + "_" + MTLNetwork.EXPERT_SCOPE, reuse=tf.AUTO_REUSE,
                                   partitioner=partitioner):
                deep_input = tf.feature_column.input_layer(features, self.deep_feature_columns)
            for i in range(self.task_num):
                with tf.variable_scope(MTLNetwork.DEEP_SCOPE + "_" + MTLNetwork.TASK_SCOPE + "_%d" % i,
                                       reuse=tf.AUTO_REUSE, partitioner=partitioner):
                    # deep_output = NNLayers.build_deep_layers(deep_input, self.expert_hidden_units, self.expert_activate_fn, self.expert_dropout, "expert")
                    deep_output = self.get_deep_layer(features, self.deep_feature_columns, self.cross_feature_columns)
                    output = NNLayers.build_deep_layers(deep_output, [1], None, 0.0, "task_out")
                    task_output.append(output)
            return task_output
        elif self.model == 'one_gate':
            expert_output = []
            task_input = 0
            task_output = []
            with tf.variable_scope(MTLNetwork.DEEP_SCOPE + "_" + MTLNetwork.EXPERT_SCOPE, reuse=tf.AUTO_REUSE,
                                   partitioner=partitioner):
                deep_input = tf.feature_column.input_layer(features, self.deep_feature_columns)
                for i in range(self.expert_num):
                    with tf.variable_scope(MTLNetwork.EXPERT_SCOPE + "_%d" % i, reuse=tf.AUTO_REUSE):
                        # deep_output = NNLayers.build_deep_layers(deep_input, self.expert_hidden_units, self.expert_activate_fn, self.expert_dropout, "expert")
                        deep_output = self.get_deep_layer(features, self.deep_feature_columns,
                                                          self.cross_feature_columns)
                        expert_output.append(deep_output)
                weight = NNLayers.build_deep_layers(deep_input, [self.expert_num], None, 0, "weight")
                weight_softmax = tf.nn.softmax(weight)
                for i in range(self.expert_num):
                    task_input += expert_output[i] * weight_softmax[:, i:i + 1]
            for i in range(self.task_num):
                with tf.variable_scope(MTLNetwork.DEEP_SCOPE + "_" + MTLNetwork.TASK_SCOPE + "_%d" % i,
                                       reuse=tf.AUTO_REUSE, partitioner=partitioner):
                    deep_output = NNLayers.build_deep_layers(task_input, self.task_hidden_units, self.task_activate_fn,
                                                             self.task_dropout, "task")
                    output = NNLayers.build_deep_layers(deep_output, [1], None, 0.0, "task_out")
                    task_output.append(output)
            return task_output
        elif self.model == 'multi_gate':
            expert_output = []
            task_input = []
            task_output = []
            with tf.variable_scope(MTLNetwork.DEEP_SCOPE + "_" + MTLNetwork.EXPERT_SCOPE, reuse=tf.AUTO_REUSE,
                                   partitioner=partitioner):
                deep_input = tf.feature_column.input_layer(features, self.deep_feature_columns)
                for i in range(self.expert_num):
                    with tf.variable_scope(MTLNetwork.EXPERT_SCOPE + "_%d" % i, reuse=tf.AUTO_REUSE,
                                           partitioner=partitioner):
                        # deep_output = NNLayers.build_deep_layers(deep_input, self.expert_hidden_units, self.expert_activate_fn, self.expert_dropout, "expert")
                        deep_output = self.get_deep_layer(features, self.deep_feature_columns,
                                                          self.cross_feature_columns)
                        expert_output.append(deep_output)
                for j in range(self.task_num):
                    with tf.variable_scope(MTLNetwork.DEEP_SCOPE + "_%d" % j, reuse=tf.AUTO_REUSE,
                                           partitioner=partitioner):
                        weight = NNLayers.build_deep_layers(deep_input, [self.expert_num], None, 0, "weight")
                        weight_softmax = tf.nn.softmax(weight)
                        tmp_input = 0
                        for i in range(len(expert_output)):
                            tmp_input += expert_output[i] * weight_softmax[:, i:i + 1]
                        task_input.append(tmp_input)
            for i in range(self.task_num):
                with tf.variable_scope(MTLNetwork.DEEP_SCOPE + "_" + MTLNetwork.TASK_SCOPE + "_%d" % i,
                                       reuse=tf.AUTO_REUSE, partitioner=partitioner):
                    deep_output = NNLayers.build_deep_layers(task_input[i], self.task_hidden_units,
                                                             self.task_activate_fn, self.task_dropout, "task")
                    output = NNLayers.build_deep_layers(deep_output, [1], None, 0.0, "task_out")
                    task_output.append(output)
            return task_output
        else:
            raise Exception("Need Implement")

    def get_deep_layer(self, features, deep_feature_columns, cross_feature_columns):
        deep_input = tf.feature_column.input_layer(features, deep_feature_columns)
        deep_output = NNLayers.build_deep_layers(deep_input, self.expert_hidden_units, self.expert_activate_fn,
                                                 self.expert_dropout, "expert")
        if self.use_cross:
            cross_input = tf.feature_column.input_layer(features, cross_feature_columns)
            if self.cross_matrix:
                cross_output = NNLayers.get_cross_matrix_layer(input_tensor=cross_input, cross_layers=self.cross_layers,
                                                               cross_dropout=self.cross_dropout, name="cross")
            else:
                cross_output = NNLayers.get_cross_layer(input_tensor=cross_input, cross_layers=self.cross_layers,
                                                        cross_dropout=self.cross_dropout, name="cross")
            deep_output = tf.concat([deep_output, cross_output], axis=-1)
        return deep_output

    def get_pre(self, ):
        return self.pre

    def get_loss(self):
        return self.loss

    def get_optimizer(self):
        return self.optimizer

    def get_metrics(self, targets, preds):
        data = {}
        for i in range(self.task_num):
            pre = self.pre[i]
            wide_pre = self.wide_pre[i]
            deep_pre = self.deep_pre[i]
            tar = targets[:, i:i + 1]
            data.update({
                "label/mean_%d" % i: tf.metrics.mean(tar),
                "prediction/mean_%d" % i: tf.metrics.mean(pre),
                "mean_absolute_error_%d" % i: tf.metrics.mean_absolute_error(tar, pre),
                "mean_squared_error_%d" % i: tf.metrics.mean_squared_error(tar, pre),
                "auc_%d" % i: tf.metrics.auc(tar, pre),
                "wide_auc_%d" % i: tf.metrics.auc(tar, wide_pre),
                "deep_auc_%d" % i: tf.metrics.auc(tar, deep_pre),
            })
        return data

    def get_eval_summary(self, targets, preds):
        summary = {}
        for key, value in self.get_metrics(targets, preds).items():
            summary[key] = value[1]
        return summary
