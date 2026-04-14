# -*- encoding:utf-8 -*-

from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import training_util
from tensorflow.python.framework import ops
from tensorflow.python.ops import state_ops
from .estimator import ModelBase
from model.nn_layers import *
from model.loss_functions import *

class WDCCNetwork(ModelBase):
    WIDE_SCOPE = "wide"
    DEEP_CROSS_SCOPE = "dnn"
    COMBINE_SCOPE = "combine"
    CAN_SCOPE = "can"
    DEFAULT_HIDDEN_UNITS = [128, 64]
    DEFAULT_WIDE_LR = 0.001
    DEFAULT_DEEP_LR = 0.003
    DEFAULT_DEEP_DROPOUT = 0.15
    DEFAULT_CROSS_DROPOUT = 0.10
    DEFAULT_COMPRESSED_DROPOUT = 0.10
    DEFAULT_CROSS_LAYERS = 3
    DEFAULT_COMPRESSED_LAYERS = [64, 32]
    DEFAULT_COMPRESSED_EMBEDDING = 8
    DEFAULT_FM_EMBEDDING = 8
    DEFAULT_ATTENTION_LAYERS = [64]
    DEFAULT_COMBINE_LR = 0.001
    DEFAULT_ORDERS = 1
    DEFAULT_WEIGHT_EMB_W = [[16, 8], [8, 4]]
    DEFAULT_WEIGHT_EMB_B = [0, 0]

    def __init__(self,
                 wide_feature_columns=None,
                 deep_feature_columns=None,
                 cross_feature_columns=None,
                 compressed_feature_columns=None,
                 seq_feature_columns=None,
                 seq_feature_size=None,
                 fm_feature_columns=None,
                 attention_list=None,
                 pnn_list=None,
                 output_dim=1,
                 use_wide=True,
                 use_cross=True,
                 use_deep=True,
                 use_compressed=False,
                 use_fm=False,
                 use_attention=False,
                 use_pnn=False,
                 pnn_inner_product=False,
                 dcn_parallel=True,
                 deep_use_bn=False,
                 cross_use_bn=False,
                 compressed_use_bn=False,
                 wide_optimizer=None,
                 deep_optimizer=None,
                 deep_dropout=DEFAULT_DEEP_DROPOUT,
                 deep_activate_fn=tf.nn.relu,
                 deep_hidden_units=None,
                 cross_dropout=DEFAULT_CROSS_DROPOUT,
                 cross_layers=DEFAULT_CROSS_LAYERS,
                 cross_l2=0.05,
                 cross_matrix=False,
                 compressed_dropout=DEFAULT_COMPRESSED_DROPOUT,
                 compressed_layers=DEFAULT_COMPRESSED_LAYERS,
                 compressed_embedding=DEFAULT_COMPRESSED_EMBEDDING,
                 compressed_file_split_half=True,
                 compressed_l2=0.05,
                 fm_embedding=DEFAULT_COMPRESSED_EMBEDDING,
                 attention_hidden_units=DEFAULT_ATTENTION_LAYERS,
                 attention_activate_fn=tf.nn.relu,
                 all_optimizer=None,
                 use_can=False,
                 can_feature_columns=None,
                 can_orders=DEFAULT_ORDERS,
                 can_weight_emb_w=DEFAULT_WEIGHT_EMB_W,
                 can_weight_emb_b=DEFAULT_WEIGHT_EMB_B,
                 use_focal_loss=False,
                 ):
        ModelBase.__init__(self, "WDCCNetwork")
        if (not use_cross) and (not use_deep) and (not use_wide) and (not use_compressed):
            raise Exception("One Unit Needed Must")
        self.wide_feature_columns = wide_feature_columns
        self.deep_feature_columns = deep_feature_columns
        self.cross_feature_columns = cross_feature_columns
        self.compressed_feature_columns = compressed_feature_columns
        self.fm_feature_columns = fm_feature_columns
        self.pnn_list = pnn_list
        self.attention_list = attention_list
        # print("attention_list", attention_list)
        self.seq_feature_columns = seq_feature_columns
        self.seq_feature_size = seq_feature_size
        self.use_wide = use_wide
        self.use_cross = use_cross
        self.use_deep = use_deep
        self.use_compressed = use_compressed
        self.use_fm = use_fm
        self.deep_use_bn = deep_use_bn
        self.cross_use_bn = cross_use_bn
        self.use_attention = use_attention
        self.use_pnn = use_pnn
        self.pnn_inner_product = pnn_inner_product
        self.compressed_use_bn = compressed_use_bn
        self.compressed_dropout = compressed_dropout
        self.wide_learning_rate = WDCCNetwork.DEFAULT_WIDE_LR
        self.deep_learning_rate = WDCCNetwork.DEFAULT_DEEP_LR
        self.output_dim = output_dim
        self.dcn_parallel = dcn_parallel
        if deep_optimizer is None:
            self.deep_optimizer = tf.train.AdagradOptimizer(
                learning_rate=self.deep_learning_rate)
        else:
            self.deep_optimizer = deep_optimizer
        if wide_optimizer is None:
            self.wide_optimizer = tf.train.FtrlOptimizer(
                learning_rate=self.wide_learning_rate)
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
        self.cross_l2 = cross_l2
        self.cross_matrix = cross_matrix
        self.compressed_layers = compressed_layers
        self.compressed_embedding = compressed_embedding
        self.compressed_file_split_half = compressed_file_split_half
        self.compressed_l2 = compressed_l2
        self.fm_embedding = fm_embedding
        self.use_attention = use_attention
        self.attention_activate_fn = attention_activate_fn
        self.attention_hidden_units = attention_hidden_units
        self.pre = None
        self.loss = None
        self.mode = None
        self.all_optimizer = all_optimizer
        self.use_can = use_can
        self.can_feature_columns = can_feature_columns
        self.can_orders = can_orders
        self.can_weight_emb_w = can_weight_emb_w
        self.can_weight_emb_b = can_weight_emb_b
        self.use_focal_loss = use_focal_loss
        self.wide_loss = 0
        self.pos_wide_pre = 0

    # features为正样本 targets为负样本
    def build_graph(self, features, targets, mode=None, config=None):
        self.mode = mode
        is_training = True
        if mode == tf.estimator.ModeKeys.PREDICT or mode == tf.estimator.ModeKeys.EVAL:
            self.deep_dropout = 0.0
            self.cross_dropout = 0.0
            is_training = False
        # 只在训练模式下进行 reshape 操作
        # targets = None
        # if self.mode == tf.estimator.ModeKeys.TRAIN or self.mode == tf.estimator.ModeKeys.EVAL:
        #     # print("targets.shape()", targets["label"].shape)
        #     targets = tf.reshape(targets["label"], (-1, 1))
        num_ps_replicas = config.num_ps_replicas if config else 0
        partitioner = partitioned_variables.min_max_variable_partitioner(
            max_partitions=num_ps_replicas,
            min_slice_size=64 << 20)
        if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
            wide_output_tensor, dnn_output_tensor, output_tensor = self.get_output_tensor(
                self.wide_feature_columns, self.deep_feature_columns, self.cross_feature_columns,
                self.compressed_feature_columns, self.fm_feature_columns, self.attention_list, self.pnn_list, features,
                partitioner, is_training)
            self.pos_pre = tf.identity(tf.nn.sigmoid(output_tensor), name="all_pre")
            if self.use_wide:
                self.pos_wide_pre = tf.identity(tf.sigmoid(wide_output_tensor), name="wide_pre")
            self.pos_deep_pre = tf.identity(tf.sigmoid(dnn_output_tensor), name="deep_pre")
            self.pre = self.pos_pre
            if self.use_focal_loss:
                self.wide_loss = tf.reduce_mean(
                    LossFunc._binary_focal_loss_from_logits(labels=targets, logits=wide_output_tensor))
                self.deep_loss = tf.reduce_mean(
                    LossFunc._binary_focal_loss_from_logits(labels=targets, logits=dnn_output_tensor))
                self.loss = tf.reduce_mean(LossFunc._binary_focal_loss_from_logits(
                    labels=targets, logits=output_tensor))
            else:
                if self.use_wide:
                    self.wide_loss = tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(labels=targets, logits=wide_output_tensor))
                self.deep_loss = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(labels=targets, logits=dnn_output_tensor))
                self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=targets, logits=output_tensor))
            self.train_op = self._train_op_fn(self.wide_loss, self.deep_loss, self.loss)
        else:
            wide_output_tensor, dnn_output_tensor, output_tensor = self.get_output_tensor(
                self.wide_feature_columns, self.deep_feature_columns, self.cross_feature_columns,
                self.compressed_feature_columns, self.fm_feature_columns, self.attention_list, self.pnn_list, features,
                partitioner, is_training)
            self.pos_pre = tf.identity(tf.nn.sigmoid(output_tensor), name="all_pre")
            if self.use_wide:
                self.pos_wide_pre = tf.identity(tf.sigmoid(wide_output_tensor), name="wide_pre")
            self.pos_deep_pre = tf.identity(tf.sigmoid(dnn_output_tensor), name="deep_pre")
            self.pre = self.pos_pre
            if targets is not None:
                if self.use_focal_loss:
                    self.loss = tf.reduce_mean(LossFunc._binary_focal_loss_from_logits(
                        labels=targets, logits=output_tensor))
                    self.wide_loss = tf.reduce_mean(LossFunc._binary_focal_loss_from_logits(
                        labels=targets, logits=wide_output_tensor))
                    self.deep_loss = tf.reduce_mean(LossFunc._binary_focal_loss_from_logits(
                        labels=targets, logits=dnn_output_tensor))
                else:
                    self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                        labels=targets, logits=output_tensor))
                    if self.use_wide:
                        self.wide_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                            labels=targets, logits=wide_output_tensor))
                    self.deep_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                        labels=targets, logits=dnn_output_tensor))
                self.train_op = self._train_op_fn(
                    self.wide_loss, self.deep_loss, self.loss)

    def cal_loss(self, predict, label):
        if self.output_dim <= 2:
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                labels=label, logits=predict))
        else:
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                labels=label, logits=predict))
        return loss

    def cal_loss_with_sigmoid(self, predict, label):
        loss = tf.reduce_mean(tf.losses.log_loss(
            labels=label, predictions=tf.clip_by_value(predict, 1e-8, 0.999999)))
        return loss

    def get_predict(self, output_tensor):
        if self.output_dim <= 2:
            pre = tf.nn.sigmoid(output_tensor)
        else:
            pre = tf.nn.softmax(output_tensor)
        return pre

    def get_output_tensor(self, wide_feature_columns, deep_feature_columns, cross_feature_columns,
                          compressed_feature_columns, fm_feature_columns, attention_list, pnn_list, features,
                          partitioner, is_training):
        wide_output_tensor = 0
        dnn_output_tensor = 0

        # wide层
        if self.use_wide:
            with tf.variable_scope(self.WIDE_SCOPE, reuse=tf.AUTO_REUSE, partitioner=partitioner):
                wide_output = tf.feature_column.linear_model(
                    features, wide_feature_columns, units=self.output_dim)
                wide_output_tensor = wide_output

        # compressed层
        if self.use_compressed:
            with tf.variable_scope(self.DEEP_CROSS_SCOPE, reuse=tf.AUTO_REUSE):
                compressed_input = tf.feature_column.input_layer(
                    features, compressed_feature_columns)
                if self.cross_use_bn:
                    compressed_input = tf.layers.batch_normalization(
                        compressed_input, training=is_training)
                compressed_input = tf.reshape(
                    compressed_input, shape=[-1, len(compressed_feature_columns), self.compressed_embedding])
                compressed_output = NNLayers.get_compressed_layer(
                    input_tensor=compressed_input, field_nums=len(
                        self.compressed_feature_columns),
                    compressed_layers=self.compressed_layers, embedding_size=self.compressed_embedding,
                    compressed_l2=self.compressed_l2, compressed_file_split_half=self.compressed_file_split_half,
                    name="compressed")
                compressed_output = NNLayers.build_deep_layers(
                    compressed_output, [self.output_dim], None, self.deep_dropout, "compressed")
                dnn_output_tensor += compressed_output

        # fm 层
        if self.use_fm:
            with tf.variable_scope(self.DEEP_CROSS_SCOPE, reuse=tf.AUTO_REUSE):
                fm_input = tf.feature_column.input_layer(
                    features, fm_feature_columns)
                fm_output = NNLayers.get_fm_layer(fm_input, self.fm_embedding)
                fm_output = NNLayers.build_deep_layers(
                    fm_output, [self.output_dim], None, self.deep_dropout, "fm")
                dnn_output_tensor += fm_output

        # din层
        attention_outputs = []
        if self.use_attention:
            with tf.variable_scope(self.DEEP_CROSS_SCOPE, reuse=tf.AUTO_REUSE):
                for index, attention_layer in enumerate(attention_list):
                    if len(attention_layer) < 2:
                        continue
                    attention_output = NNLayers.get_custom_attention_layer(
                        attention_layer, features=features, hidden=self.attention_hidden_units,
                        name="%s_%d" % ("attention", index))
                    attention_outputs.append(attention_output)

        # # seqs 层
        # seq_outputs = []
        # if self.seq_feature_columns:
        #     for index, one in enumerate(self.seq_feature_columns):
        #         embedding, length = sequence_input_layer(features, [one])
        #         shape = embedding.get_shape().as_list()
        #         embedding = tf.reshape(
        #             embedding, [-1, self.seq_feature_size[index] * shape[2]])
        #         seq_outputs.append(embedding)

        # pnn层
        pnn_outputs = []
        if self.use_pnn:
            with tf.variable_scope(self.DEEP_CROSS_SCOPE, reuse=tf.AUTO_REUSE):
                for pnn_pair in pnn_list:
                    if len(pnn_pair) == 0:
                        continue
                    root_vec = tf.feature_column.input_layer(
                        features, [pnn_pair[-1]])
                    for i in range(len(pnn_pair) - 1):
                        child_vec = tf.feature_column.input_layer(
                            features, [pnn_pair[i]])
                        tmp_output = NNLayers.get_cosine_layer(
                            root_vec, child_vec, inner_product=self.pnn_inner_product)
                        pnn_outputs.append(tmp_output)

        # cross层
        cross_outputs = None
        if self.use_cross:
            with tf.variable_scope(self.DEEP_CROSS_SCOPE, reuse=tf.AUTO_REUSE):
                cross_input = tf.feature_column.input_layer(
                    features, cross_feature_columns)
                if self.cross_use_bn:
                    cross_input = tf.layers.batch_normalization(
                        cross_input, training=is_training)
                if self.cross_matrix:
                    cross_output = NNLayers.get_cross_matrix_layer(input_tensor=cross_input,
                                                                   cross_layers=self.cross_layers,
                                                                   cross_dropout=self.cross_dropout, name="cross")
                else:
                    cross_output = NNLayers.get_cross_layer(input_tensor=cross_input, cross_layers=self.cross_layers,
                                                            cross_dropout=self.cross_dropout, name="cross")
                if self.dcn_parallel:
                    cross_output = NNLayers.build_deep_layers(
                        cross_output, [self.output_dim], None, 0.0, "cross")
                    dnn_output_tensor += cross_output

        # deep层
        if self.use_deep:
            with tf.variable_scope(self.DEEP_CROSS_SCOPE, reuse=tf.AUTO_REUSE):
                deep_input = tf.feature_column.input_layer(
                    features, deep_feature_columns)
                deep_inputs = [deep_input]
                if cross_outputs is not None and (not self.dcn_parallel) and self.use_cross:
                    deep_inputs.append(cross_output)
                if len(pnn_outputs) > 0 and self.use_pnn:
                    deep_inputs.extend(pnn_outputs)
                if len(attention_outputs) > 0 and self.use_attention:
                    deep_inputs.extend(attention_outputs)
                deep_input = tf.concat(deep_inputs, axis=-1)
                if self.deep_use_bn:
                    deep_input = tf.layers.batch_normalization(
                        deep_input, training=is_training)
                deep_output = NNLayers.build_deep_layers(
                    deep_input, self.deep_hidden_units, self.deep_activate_fn, self.deep_dropout, "deep_cross")
                deep_output = NNLayers.build_deep_layers(
                    deep_output, [self.output_dim], None, 0.0, "deep_cross_out")

                dnn_output_tensor += deep_output
        if self.use_wide:
            output_tensor = tf.add(wide_output_tensor, dnn_output_tensor)
        else:
            output_tensor = dnn_output_tensor
        return wide_output_tensor, dnn_output_tensor, output_tensor

    def _train_op_fn(self, wide_loss, dnn_loss, all_loss):
        train_ops = []
        global_step = training_util.get_global_step()
        if self.all_optimizer:
            train_ops.append(self.all_optimizer.minimize(
                all_loss, global_step=global_step, var_list=ops.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES)))
        else:
            if self.use_cross or self.use_deep or self.use_attention or self.use_fm:
                train_ops.append(self.deep_optimizer.minimize(
                    all_loss, global_step=global_step,
                    var_list=ops.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES, scope=self.DEEP_CROSS_SCOPE)))
            if self.use_wide:
                train_ops.append(self.wide_optimizer.minimize(
                    all_loss, global_step=global_step,
                    var_list=ops.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES, scope=self.WIDE_SCOPE)))
        train_op = control_flow_ops.group(*train_ops)
        with ops.control_dependencies([train_op]):
            with ops.colocate_with(global_step):
                return state_ops.assign_add(global_step, 1)

    def get_pre(self, ):
        return [self.pos_deep_pre, self.pre]

    def get_loss(self):
        return self.loss

    def get_optimizer(self):
        return self.optimizer

    def get_metrics(self, targets, all_preds):
        deep_pre, all_pre = all_preds
        # label = tf.reshape(targets["label"], (-1, 1))
        return {
            'label/mean': tf.metrics.mean(targets),
            'prediction/mean': tf.metrics.mean(all_pre),
            "mean_absolute_error": tf.metrics.mean_absolute_error(targets, all_pre),
            "mean_squared_error": tf.metrics.mean_squared_error(targets, all_pre),
            # "loss": tf.metrics.mean(self.loss),
            'all_auc': tf.metrics.auc(targets, all_pre),
            'deep_auc': tf.metrics.auc(targets, deep_pre),
            # 'wide_auc': tf.metrics.auc(targets, wide_pre),
        }

    def get_eval_summary(self, targets, all_preds):
        summary = {}
        for key, value in self.get_metrics(targets, all_preds).items():
            summary[key] = value[1]
        return summary
