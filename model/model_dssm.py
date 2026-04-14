# -*- encoding:utf-8 -*-

from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import training_util
from tensorflow.python.framework import ops
from tensorflow.python.ops import state_ops
from .estimator import ModelBase
from model.nn_layers import *
from model.contrib.losses import *

class DSSMNetwork(ModelBase):
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
                 user_feature_columns=None,
                 item_feature_columns=None,
                 user_key_columns=None,
                 item_key_columns=None,
                 compressed_feature_columns=None,
                 seq_feature_columns=None,
                 seq_feature_size=None,
                 fm_feature_columns=None,
                 attention_list=None,
                 pnn_list=None,
                 output_dim=1,
                 vec_dim=32,
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
                 use_focal_loss=False,
                 temp=0.1
                 ):
        ModelBase.__init__(self, "DSSMNetwork")
        if (not use_cross) and (not use_deep) and (not use_compressed):
            raise Exception("One Unit Needed Must")
        self.wide_feature_columns = wide_feature_columns
        self.deep_feature_columns = deep_feature_columns
        self.cross_feature_columns = cross_feature_columns
        self.user_feature_columns = user_feature_columns
        self.item_feature_columns = item_feature_columns
        self.user_key_columns = user_key_columns
        self.item_key_columns = item_key_columns
        self.compressed_feature_columns = compressed_feature_columns
        self.fm_feature_columns = fm_feature_columns
        self.pnn_list = pnn_list
        self.attention_list = attention_list
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
        self.wide_learning_rate = DSSMNetwork.DEFAULT_WIDE_LR
        self.deep_learning_rate = DSSMNetwork.DEFAULT_DEEP_LR
        self.output_dim = output_dim
        self.vec_dim = vec_dim
        self.dcn_parallel = dcn_parallel
        self.temp = temp
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

    def get_output_tensor(self, deep_feature_columns, key_feature_columns, features, is_training, name):
        with tf.variable_scope("shortcut" + name, reuse=tf.AUTO_REUSE):
            short_input = tf.feature_column.input_layer(features, key_feature_columns)
            tower_shortcut = K.layers.Dense(self.deep_hidden_units[-1])(short_input)
            # with tf.variable_scope(self.DEEP_CROSS_SCOPE + name, reuse=tf.AUTO_REUSE):
            deep_input = tf.feature_column.input_layer(features, deep_feature_columns)
            deep_output = NNLayers.build_deep_layers_bn(deep_input, self.deep_hidden_units[:-1], 'relu',
                                                        is_training, "deep1_" + name)
            deep_output = NNLayers.build_deep_layers_bn(deep_output, [self.deep_hidden_units[-1]], None,
                                                        is_training, "deep2_" + name)
            user_tower = tf.add(deep_output, tower_shortcut)
            dnn_output_tensor = tf.math.tanh(user_tower)

        return dnn_output_tensor

    def build_graph(self, features, targets, mode=None, config=None):
        self.mode = mode
        is_training = True
        if mode == tf.estimator.ModeKeys.PREDICT or mode == tf.estimator.ModeKeys.EVAL:
            self.deep_dropout = 0.0
            self.cross_dropout = 0.0
            is_training = False

        # num_ps_replicas = config.num_ps_replicas if config else 0
        # partitioner = partitioned_variables.min_max_variable_partitioner(
        #     max_partitions=num_ps_replicas,
        #     min_slice_size=64 << 20)
        # targets1 = None
        # if self.mode == tf.estimator.ModeKeys.TRAIN or self.mode == tf.estimator.ModeKeys.EVAL:
        #     # print("targets.shape()", targets["label"].shape)
        #     targets1 = tf.reshape(targets["label"], (-1, 1))
        user_embedding = self.get_output_tensor(self.user_feature_columns, self.user_key_columns,
                                                features, is_training, "user")
        item_embedding = self.get_output_tensor(self.item_feature_columns, self.item_key_columns,
                                                features, is_training, "item")
        # [batch, K]
        user_embedding_norm = tf.math.l2_normalize(user_embedding, axis=1, name="user_embedding_norm")
        # [batch, K]
        item_embedding_norm = tf.math.l2_normalize(item_embedding, axis=1, name="item_embedding_norm")
        if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
            nce_loss, outer_product, nce_numerator, _ = compute_masked_info_nce_loss(user_embedding_norm,
                                                                                     item_embedding_norm, self.temp)
            self.loss = tf.reduce_mean(nce_loss)
            # self.pre = tf.nn.sigmoid(user_embedding_norm * item_embedding_norm)
            # logits = tf.identity(
            #     tf.reduce_mean(tf.clip_by_value(tf.multiply(user_embedding_norm, item_embedding_norm), -500, 500),
            #                    keepdims=True, axis=1), name="logit")
            self.pre = [tf.identity(nce_numerator, name="pre"),
                        tf.identity(user_embedding_norm, name='user_embedding'),
                        tf.identity(item_embedding_norm, name='item_embedding'),
                        outer_product]
            self.train_op = self._train_op_fn(self.loss)
        elif mode == tf.estimator.ModeKeys.PREDICT:
            nce_loss, outer_product, nce_numerator, _ = compute_masked_info_nce_loss(user_embedding_norm,
                                                                                     item_embedding_norm, self.temp)
            # logits = tf.identity(
            #     tf.reduce_mean(tf.clip_by_value(tf.multiply(user_embedding_norm, item_embedding_norm), -500, 500),
            #                    keepdims=True, axis=1), name="logit")
            self.pre = [tf.identity(nce_numerator, name="pre"),
                        tf.identity(user_embedding_norm, name='user_embedding'),
                        tf.identity(item_embedding_norm, name='item_embedding'),
                        outer_product]

    def _train_op_fn(self, all_loss):
        train_ops = []
        global_step = training_util.get_global_step()
        train_ops.append(self.all_optimizer.minimize(
            all_loss, global_step=global_step,
            var_list=ops.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES)))
        train_op = control_flow_ops.group(*train_ops)
        with ops.control_dependencies([train_op]):
            with ops.colocate_with(global_step):
                return state_ops.assign_add(global_step, 1)

    def get_pre(self, ):
        return self.pre

    def get_loss(self):
        return self.loss

    def get_optimizer(self):
        return self.optimizer

    def get_metrics(self, targets, pred):
        # label = tf.reshape(targets["label"], (-1, 1))
        pre, _, _, outer_product = pred
        batch = tf.shape(outer_product, out_type=tf.dtypes.int64, name="current_batch_size")[0]
        recall_10 = tf.cast(tf.math.in_top_k(outer_product, tf.range(batch), k=10, name="recall_at_10"), tf.float32)
        recall_50 = tf.cast(tf.math.in_top_k(outer_product, tf.range(batch), k=50, name="recall_at_50"), tf.float32)
        recall_100 = tf.cast(tf.math.in_top_k(outer_product, tf.range(batch), k=100, name="recall_at_100"), tf.float32)
        return {
            'label/mean': tf.metrics.mean(targets),
            'prediction/mean': tf.metrics.mean(pre),
            "mean_absolute_error": tf.metrics.mean_absolute_error(targets, pre),
            "mean_squared_error": tf.metrics.mean_squared_error(targets, pre),
            "recall_10": tf.metrics.mean(recall_10),
            "recall_50": tf.metrics.mean(recall_50),
            "recall_100": tf.metrics.mean(recall_100),
        }

    def get_eval_summary(self, targets, pred):
        summary = {}
        for key, value in self.get_metrics(targets, pred).items():
            summary[key] = value[1]
        return summary
