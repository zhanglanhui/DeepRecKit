#!/usr/bin/python
# -*- encoding: utf-8 -*-


import copy
import tensorflow as tf
from tensorflow.keras.backend import expand_dims, repeat_elements, sum
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import training_util
from tensorflow.python.framework import ops
from tensorflow.python.ops import state_ops
from .estimator import ModelBase
from model.nn_layers import NNLayers
from model.loss_functions import LossFunc

def dense_layer(i, units, activation, name, weight_name):
    with tf.name_scope(name):
        rows = tf.shape(i)[0]
        weight = tf.get_variable(weight_name, (i.get_shape()[1] + 1, units))
        bias_input = tf.fill([rows, 1], 1.0, name='bias_input')
        o = tf.matmul(tf.concat([i, bias_input], 1), weight, name=name + '_mul')
        if activation is not None:
            return activation(o)
        else:
            return o

def simple_dense_network(inputs, units, name, weight_name_template, act=tf.nn.relu):
    output = inputs
    for i, unit in enumerate(units):
        output = dense_layer(output, unit, act, name='dense_{}_{}'.format(name, i),
                             weight_name=weight_name_template.format(i + 1))
    return output

def simple_lhuc_network(inputs, unit1, unit2, name, weight_name):
    with tf.name_scope('{}_lhuc'.format(name)):
        output = inputs
        with tf.name_scope('{}_lhuc_layer_{}'.format(name, 0)):
            output = dense_layer(output, unit1, tf.nn.relu, name='dense_{}_{}'.format(name, 0),
                                 weight_name='{}_layer1_param'.format(weight_name))
        with tf.name_scope('{}_lhuc_layer_{}'.format(name, 1)):
            output = 2.0 * dense_layer(output, unit2, tf.nn.sigmoid, name='dense_{}_{}'.format(name, 1),
                                       weight_name='{}_layer2_param'.format(weight_name))
        return output

def mmoe_layer(inputs, expert_units, name, num_experts, num_tasks, expert_act=tf.nn.relu, gate_act=tf.nn.softmax):
    expert_outputs, final_outputs = [], []
    with tf.name_scope('experts_network'):
        for i in range(num_experts):
            weight_name_template = name + '_expert{}_'.format(i) + '_h{}_param'
            expert_layer = simple_dense_network(inputs, expert_units, '{}_experts'.format(name),
                                                weight_name_template, act=expert_act)
            expert_outputs.append(expand_dims(expert_layer, axis=2))
        expert_outputs = tf.concat(expert_outputs, 2)

    with tf.name_scope('gates_network'):
        for i in range(num_tasks):
            weight_name_template = name + '_task_gate{}_'.format(i) + 'param'
            gate_layer = dense_layer(inputs, num_experts, gate_act, '{}_gates'.format(name), weight_name_template)
            expanded_gate_output = expand_dims(gate_layer, axis=1)
            weighted_expert_output = expert_outputs * repeat_elements(expanded_gate_output, expert_units[-1], axis=1)
            final_outputs.append(sum(weighted_expert_output, axis=2))

def build_concat_tower_logits(tower1, tower2, concat_name, concat_tower_units):
    concat_tower_input = tf.concat([tower1, tower2], axis=-1)
    concat_tower_output = NNLayers.build_deep_layers(
        concat_tower_input, concat_tower_units, tf.nn.relu, dropout=0.005, name=concat_name)
    # concat_tower_output_layer = tf.layers.dense(
    #     concat_tower_module,
    #     units=self.concat_tower_units[-1],
    #     activation=None,
    #     kernel_initializer=tf.glorot_uniform_initializer(),
    #     name=concat_name+"_concat_tower_output_layer"
    # )
    # concat_tower_output = tf.identity(
    #     concat_tower_output_layer, name="{0}_concat_tower_output".format(concat_name))
    return concat_tower_output

class MTLMFHNetWork(ModelBase):
    def __init__(self,
                 common_dense_input_features,
                 mfh1_dense_input_features,
                 mfh2_dense_input_features,
                 common_sparse_input_features,
                 weight_input_features,
                 mfh_mask_features,
                 mfh_names,
                 task_names,
                 expert_units=[64],
                 task_units=[64],
                 concat_tower_units=[64],
                 task1_dense_optimizer=None,
                 task2_dense_optimizer=None,
                 task3_dense_optimizer=None,
                 sparse_optimizer=None,
                 with_sparse_optimizer=False,
                 use_focal_loss=False,
                 ):
        ModelBase.__init__(self, "MTLMFHNetWork")
        self.common_dense_input_features = common_dense_input_features
        self.mfh1_dense_input_features = mfh1_dense_input_features
        self.mfh2_dense_input_features = mfh2_dense_input_features
        self.common_sparse_input_features = common_sparse_input_features
        self.weight_input_features = weight_input_features
        self.mfh_mask_features = mfh_mask_features
        self.mfh_names = mfh_names
        self.task_names = task_names
        self.expert_units = expert_units
        self.task_units = task_units
        self.concat_tower_units = concat_tower_units
        self.with_sparse_optimizer = with_sparse_optimizer
        self.use_focal_loss = use_focal_loss
        self.pre = None
        if task1_dense_optimizer is None:
            self.task1_dense_optimizer = tf.train.AdagradOptimizer(learning_rate=0.005)
        else:
            self.task1_dense_optimizer = task1_dense_optimizer
        if task2_dense_optimizer is None:
            self.task2_dense_optimizer = tf.train.AdagradOptimizer(learning_rate=0.005)
        else:
            self.task2_dense_optimizer = task2_dense_optimizer
        if task3_dense_optimizer is None:
            self.task3_dense_optimizer = tf.train.AdagradOptimizer(learning_rate=0.005)
        else:
            self.task3_dense_optimizer = task2_dense_optimizer
        if sparse_optimizer is None:
            self.sparse_optimizer = tf.train.FtrlOptimizer(learning_rate=0.001)
        else:
            self.sparse_optimizer = sparse_optimizer
        self.with_sparse_optimizer = with_sparse_optimizer
        self.task1_dense_loss = None
        self.task2_dense_loss = None
        self.task3_dense_loss = None
        self.sparse_pre = None
        self.print_ops = []

    def build_graph(self, features, targets, mode=None, config=None):
        with tf.variable_scope("sparse", reuse=tf.AUTO_REUSE):
            sparse_out = tf.feature_column.linear_model(features, self.common_sparse_input_features, 1, "sum")
        with tf.variable_scope("dense", reuse=tf.AUTO_REUSE):
            common_dense_input = tf.feature_column.input_layer(features, self.common_dense_input_features)
            mfh1_dense_input = tf.feature_column.input_layer(features, self.mfh1_dense_input_features)
            mfh2_dense_input = tf.feature_column.input_layer(features, self.mfh2_dense_input_features)
            mfh1_dense_input = tf.concat([mfh1_dense_input, common_dense_input], axis=-1)
            mfh2_dense_input = tf.concat([mfh2_dense_input, common_dense_input], axis=-1)
            expert_outputs = []
            with tf.variable_scope('experts_network', reuse=tf.AUTO_REUSE):
                name = 'expert1'
                weight_name = name + "_weight"
                expert_layer = simple_dense_network(mfh1_dense_input, self.expert_units, name, weight_name,
                                                    act=tf.nn.relu)
                expert_outputs.append(expand_dims(expert_layer, axis=2))
                name = 'expert2'
                weight_name = name + "_weight"
                expert_layer = simple_dense_network(mfh2_dense_input, self.expert_units, name, weight_name,
                                                    act=tf.nn.relu)
                expert_outputs.append(expand_dims(expert_layer, axis=2))
            expert_outputs = tf.concat(expert_outputs, 2)

            mfh_label = tf.feature_column.input_layer(features, self.mfh_mask_features)
            # self.print_ops.append(tf.print("[DEBUG] ============= mfh_label: ", mfh_label))
            # mask
            mfh_mask1 = tf.cast(tf.equal(tf.cast(mfh_label, tf.int8), 0), tf.float32)
            mfh_mask2 = tf.cast(tf.equal(tf.cast(mfh_label, tf.int8), 1), tf.float32)
            dense_input = tf.where_v2(tf.equal(tf.cast(mfh_label, tf.int8), 1), mfh2_dense_input, mfh1_dense_input)
            final_outputs = []
            gate_input = tf.stop_gradient(dense_input)
            with tf.variable_scope('gates_network', reuse=tf.AUTO_REUSE):
                for i in range(3 + 2):
                    gate_layer = dense_layer(gate_input, 2, tf.nn.softmax, 'gate_%d' % i, 'gate_weight_%d' % i)
                    expanded_gate_output = expand_dims(gate_layer, axis=1)
                    weighted_expert_output = expert_outputs * repeat_elements(expanded_gate_output,
                                                                              self.expert_units[-1], axis=1)
                    final_outputs.append(sum(weighted_expert_output, axis=2))

            task_units = copy.deepcopy(self.task_units)
            # build task tower
            with tf.variable_scope("task_network", reuse=tf.AUTO_REUSE):
                task1_tower = simple_dense_network(final_outputs[0], task_units, "task1", "task1_weight", tf.nn.relu)
                task2_tower = simple_dense_network(final_outputs[1], task_units, "task2", "task2_weight", tf.nn.relu)
                task3_tower = simple_dense_network(final_outputs[2], task_units, "task3", "task3_weight", tf.nn.relu)

            concat_tower_units = copy.deepcopy(self.concat_tower_units)
            # build mfh tower
            with tf.variable_scope("mfh_network", reuse=tf.AUTO_REUSE):
                mfh1_tower = simple_dense_network(final_outputs[3], task_units, "mfh0", "mfh0_weight", tf.nn.relu)
                mfh2_tower = simple_dense_network(final_outputs[4], task_units, "mfh1", "mfh1_weight", tf.nn.relu)
                tower_logit1 = build_concat_tower_logits(mfh1_tower, task1_tower, "mfh1_task1", concat_tower_units)
                tower_logit2 = build_concat_tower_logits(mfh1_tower, task2_tower, "mfh1_task2", concat_tower_units)
                tower_logit3 = build_concat_tower_logits(mfh1_tower, task3_tower, "mfh1_task3", concat_tower_units)
                tower_logit4 = build_concat_tower_logits(mfh2_tower, task1_tower, "mfh2_task1", concat_tower_units)
                tower_logit5 = build_concat_tower_logits(mfh2_tower, task2_tower, "mfh2_task2", concat_tower_units)
                tower_logit6 = build_concat_tower_logits(mfh2_tower, task3_tower, "mfh2_task3", concat_tower_units)

                mfh1_task1_out = dense_layer(tower_logit1, 1, None, "mfh1_task1_out", "mfh1_task1_out_weight")
                mfh1_task2_out = dense_layer(tower_logit2, 1, None, "mfh1_task2_out", "mfh1_task2_out_weight")
                mfh1_task3_out = dense_layer(tower_logit3, 1, None, "mfh1_task3_out", "mfh1_task3_out_weight")
                mfh2_task1_out = dense_layer(tower_logit4, 1, None, "mfh2_task1_out", "mfh2_task1_out_weight")
                mfh2_task2_out = dense_layer(tower_logit5, 1, None, "mfh2_task2_out", "mfh2_task2_out_weight")
                mfh2_task3_out = dense_layer(tower_logit6, 1, None, "mfh2_task3_out", "mfh2_task3_out_weight")

            task1_out = tf.where_v2(tf.equal(tf.cast(mfh_label, tf.int8), 1), mfh2_task1_out, mfh1_task1_out)
            task2_out = tf.where_v2(tf.equal(tf.cast(mfh_label, tf.int8), 1), mfh2_task2_out, mfh1_task2_out)
            task3_out = tf.where_v2(tf.equal(tf.cast(mfh_label, tf.int8), 1), mfh2_task3_out, mfh1_task3_out)

            task1_pre = tf.nn.sigmoid(task1_out)
            task2_pre = tf.nn.sigmoid(task2_out)
            task3_pre = tf.nn.sigmoid(task3_out)
            self.sparse_pre = tf.nn.sigmoid(sparse_out)
        self.pre = [
            tf.identity(task1_pre, name="task1_pre"),
            tf.identity(task2_pre, name="task2_pre"),
            tf.identity(task3_pre, name="task3_pre"),
        ]
        if targets is not None:
            weight_input = tf.feature_column.input_layer(features, self.weight_input_features)
            mask = tf.where_v2(tf.equal(tf.cast(mfh_label, tf.int8), 1), mfh_mask2, mfh_mask1)

            # self.print_ops.append(tf.print("[DEBUG] ============= mask1, mask2: ", mfh_mask1, mfh_mask2))
            # mask_idx1 = tf.where(tf.squeeze(mfh_mask1, -1)) #batch,1
            # mask_idx1 = tf.squeeze(mask_idx1, -1) #batch,
            # mask_idx2 = tf.where(tf.squeeze(mfh_mask2, -1)) #batch,1
            # mask_idx2 = tf.squeeze(mask_idx2, -1) #batch,
            # mfh1_task1_pred_to_append = tf.gather(task1_out, mask_idx1) 
            # mfh2_task1_pred_to_append = tf.gather(task1_out, mask_idx2)
            # mfh1_task1_label_to_append = tf.gather(targets[:, 0:1], mask_idx1)
            # mfh2_task1_label_to_append = tf.gather(targets[:, 0:1], mask_idx2)
            # mfh1_task2_pred_to_append = tf.gather(task2_out, mask_idx1) 
            # mfh2_task2_pred_to_append = tf.gather(task2_out, mask_idx2)
            # mfh1_task2_label_to_append = tf.gather(targets[:, 1:2], mask_idx1)
            # mfh2_task2_label_to_append = tf.gather(targets[:, 1:2], mask_idx2)
            # mfh1_task3_pred_to_append = tf.gather(task3_out, mask_idx1) 
            # mfh2_task3_pred_to_append = tf.gather(task3_out, mask_idx2)
            # mfh1_task3_label_to_append = tf.gather(targets[:, 2:3], mask_idx1)
            # mfh2_task3_label_to_append = tf.gather(targets[:, 2:3], mask_idx2)
            # self.print_ops.append(tf.print("[DEBUG] ============= mask_idx1, mask_idx2, mfh1_task1_pred_to_append, mfh1_task1_label_to_append: ", mask_idx1, mask_idx2, mfh1_task1_pred_to_append, mfh1_task1_label_to_append))
            # self.print_ops.append(tf.print("[DEBUG] ============= mask_idx1, mask_idx2, mfh2_task1_pred_to_append, mfh2_task1_label_to_append: ", mask_idx1, mask_idx2, mfh2_task1_pred_to_append, mfh2_task1_label_to_append))
            # self.print_ops.append(tf.print("[DEBUG] ============= mask_idx1, mask_idx2, mfh1_task2_pred_to_append, mfh1_task2_label_to_append: ", mask_idx1, mask_idx2, mfh1_task2_pred_to_append, mfh1_task2_label_to_append))
            # self.print_ops.append(tf.print("[DEBUG] ============= mask_idx1, mask_idx2, mfh2_task2_pred_to_append, mfh2_task2_label_to_append: ", mask_idx1, mask_idx2, mfh2_task2_pred_to_append, mfh2_task2_label_to_append))
            # self.print_ops.append(tf.print("[DEBUG] ============= mask_idx1, mask_idx2, mfh1_task3_pred_to_append, mfh1_task3_label_to_append: ", mask_idx1, mask_idx2, mfh1_task3_pred_to_append, mfh1_task3_label_to_append))
            # self.print_ops.append(tf.print("[DEBUG] ============= mask_idx1, mask_idx2, mfh2_task3_pred_to_append, mfh2_task3_label_to_append: ", mask_idx1, mask_idx2, mfh2_task3_pred_to_append, mfh2_task3_label_to_append))
            # self.task1_loss = self.cal_loss(predicts=[mfh1_task1_pred_to_append, mfh2_task1_pred_to_append], labels=[mfh1_task1_label_to_append, mfh2_task1_label_to_append]) 
            # self.task2_loss = self.cal_loss(predicts=[mfh1_task2_pred_to_append, mfh2_task2_pred_to_append], labels=[mfh1_task2_label_to_append, mfh2_task2_label_to_append])
            # self.task3_loss = self.cal_loss(predicts=[mfh1_task3_pred_to_append, mfh2_task3_pred_to_append], labels=[mfh1_task3_label_to_append, mfh2_task3_label_to_append]) 
            self.task1_loss = self.cal_loss_with_mask(predicts=task1_out, labels=targets[:, 0:1], mask=mask)
            self.task2_loss = self.cal_loss_with_mask(predicts=task2_out, labels=targets[:, 1:2], mask=mask)
            self.task3_loss = self.cal_loss_with_mask(predicts=task3_out, labels=targets[:, 2:3], mask=mask)
            self.print_ops.append(tf.print("[DEBUG] ============= task1_loss: ", self.task1_loss))
            self.print_ops.append(tf.print("[DEBUG] ============= task2_loss: ", self.task2_loss))
            self.print_ops.append(tf.print("[DEBUG] ============= task3_loss: ", self.task3_loss))

            self.loss = self.task1_loss + self.task2_loss + self.task3_loss
            self.train_op = self._train_op_fn(self.task1_loss, self.task2_loss, self.task3_loss)

    def cal_loss_with_mask(self, predicts, labels, mask):
        # self.print_ops.append(tf.print("[DEBUG] ============= mask: ", mask))
        mask_idx = tf.where(tf.squeeze(mask, -1))  # batch,1
        mask_idx = tf.squeeze(mask_idx, -1)  # batch,
        label = tf.gather(labels, mask_idx)
        predict = tf.gather(predicts, mask_idx)
        # self.print_ops.append(tf.print("[DEBUG] ============= labels_to_append, pred_to_append: ", label, predict))
        if self.use_focal_loss:
            loss = LossFunc._binary_focal_loss_from_logits(labels=label, logits=predict)
        else:
            loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=predict)
            # self.print_ops.append(tf.print("[DEBUG] ============= loss before reduce_mean: ", loss))
        loss = tf.reduce_mean(loss * mask)
        # self.print_ops.append(tf.print("[DEBUG] ============= loss sfter reduce_mean: ", loss))
        return loss

    # def cal_loss(self, predicts, labels):
    #     loss = 0
    #     for i in range(len(predicts)):
    #         predict = predicts[i]
    #         label = predicts[i]
    #         if self.use_focal_loss:
    #             tmp_loss = LossFunc._binary_focal_loss_from_logits(labels=label, logits=predict)
    #         else:
    #             self.print_ops.append(tf.print("[DEBUG] ============= sigmoid_cross_entropy_with_logits: ", tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=predict)))
    #             tmp_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=predict)
    #         self.print_ops.append(tf.print("[DEBUG] ============= tmp_loss before transform: ", tmp_loss))
    #         tmp_size = tf.size(tmp_loss)
    #         tmp_loss = tf.cond(tf.equal(tf.size(tmp_loss), 0), lambda: tf.zeros([2,tmp_size]), lambda: tmp_loss)
    #         self.print_ops.append(tf.print("[DEBUG] ============= tmp_loss after transform: ", tmp_loss))
    #     loss += tmp_loss
    #     self.print_ops.append(tf.print("[DEBUG] ============= loss before reduce_mean: ", loss))
    #     loss = tf.reduce_mean(loss)
    #     self.print_ops.append(tf.print("[DEBUG] ============= loss after reduce_mean: ", loss))
    #     return loss

    def _train_op_fn(self, task1_loss, task2_loss, task3_loss):
        train_ops = []
        global_step = training_util.get_global_step()
        train_ops.append(self.task1_dense_optimizer.minimize(task1_loss, global_step=global_step,
                                                             var_list=ops.get_collection(
                                                                 ops.GraphKeys.TRAINABLE_VARIABLES, scope="dense")))
        train_ops.append(self.task2_dense_optimizer.minimize(task2_loss, global_step=global_step,
                                                             var_list=ops.get_collection(
                                                                 ops.GraphKeys.TRAINABLE_VARIABLES, scope="dense")))
        train_ops.append(self.task3_dense_optimizer.minimize(task3_loss, global_step=global_step,
                                                             var_list=ops.get_collection(
                                                                 ops.GraphKeys.TRAINABLE_VARIABLES, scope="dense")))
        if self.with_sparse_optimizer:
            train_ops.append(self.sparse_optimizer.minimize(task1_loss, global_step=global_step,
                                                            var_list=ops.get_collection(
                                                                ops.GraphKeys.TRAINABLE_VARIABLES, scope="sparse")))
            train_ops.append(self.sparse_optimizer.minimize(task1_loss, global_step=global_step,
                                                            var_list=ops.get_collection(
                                                                ops.GraphKeys.TRAINABLE_VARIABLES, scope="sparse")))
            train_ops.append(self.sparse_optimizer.minimize(task1_loss, global_step=global_step,
                                                            var_list=ops.get_collection(
                                                                ops.GraphKeys.TRAINABLE_VARIABLES, scope="sparse")))
        train_op = control_flow_ops.group(*train_ops)
        print_op = control_flow_ops.group(*self.print_ops)
        with ops.control_dependencies([train_op, print_op]):
            with ops.colocate_with(global_step):
                return state_ops.assign_add(global_step, 1)

    def get_pre(self, ):
        return self.pre

    def get_loss(self):
        return self.loss

    def get_metrics(self, targets, preds):
        task1_pre, task2_pre, task3_pre = preds
        task1_label, task2_label, task3_label = targets[:, 0:1], targets[:, 1:2], targets[:, 2:3]
        return {
            'task1_label/mean': tf.metrics.mean(task1_label),
            'task1_prediction/mean': tf.metrics.mean(task1_pre),
            "task1_mean_absolute_error": tf.metrics.mean_absolute_error(task1_label, task1_pre),
            "task1_mean_squared_error": tf.metrics.mean_squared_error(task1_label, task1_pre),
            'task1_auc': tf.metrics.auc(task1_label, task1_pre),
            'task1_wide_auc': tf.metrics.auc(task1_label, self.sparse_pre),
            "task1_loss": tf.metrics.mean(self.task1_loss),

            'task2_label/mean': tf.metrics.mean(task2_label),
            'task2_prediction/mean': tf.metrics.mean(task2_pre),
            "task2_mean_absolute_error": tf.metrics.mean_absolute_error(task2_label, task2_pre),
            "task2_mean_squared_error": tf.metrics.mean_squared_error(task2_label, task2_pre),
            'task2_auc': tf.metrics.auc(task2_label, task2_pre),
            'task2_wide_auc': tf.metrics.auc(task2_label, self.sparse_pre),
            "task2_loss": tf.metrics.mean(self.task2_loss),

            'task3_label/mean': tf.metrics.mean(task3_label),
            'task3_prediction/mean': tf.metrics.mean(task3_pre),
            "task3_mean_absolute_error": tf.metrics.mean_absolute_error(task3_label, task3_pre),
            "task3_mean_squared_error": tf.metrics.mean_squared_error(task3_label, task3_pre),
            'task3_auc': tf.metrics.auc(task3_label, task3_pre),
            'task2_wide_auc': tf.metrics.auc(task3_label, self.sparse_pre),
            "task3_loss": tf.metrics.mean(self.task3_loss),
        }

    def get_eval_summary(self, targets, preds):
        summary = {}
        for key, value in self.get_metrics(targets, preds).items():
            summary[key] = value[1]
        return summary
