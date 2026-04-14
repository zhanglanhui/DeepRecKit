# -*- encoding:utf-8 -*-
import copy
import tensorflow as tf
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.training import training_util
from tensorflow.python.feature_column.feature_column import _CategoricalColumn
from tensorflow.python.feature_column.feature_column import _LazyBuilder
from features.embedding_lookup import safe_embedding_lookup_sparse
from tensorflow.python.ops import state_ops
from .estimator import ModelBase

WIDE_SCOPE = "wide"
DEEPFM_SCOPE = "deepfm"

class DeepFM(ModelBase):
    def __init__(self,
                 fm_feature_columns,
                 wide_feature_columns,
                 deep_feature_columns,
                 embedding_size=32,
                 hidden_units=[64],
                 learning_rate=0.005,
                 output_dim=1,
                 activate_fn=tf.nn.relu,
                 log_loss=True,
                 wide_optimizer=None,
                 deepfm_optimizer=None,
                 dropout_fm=0.0,
                 dropout_deep=0.0,
                 regression=False,
                 share_embedding_dict=None):
        ModelBase.__init__(self, "DeepFM")
        self.fm_feature_columns = fm_feature_columns
        self.wide_feature_columns = wide_feature_columns
        self.deep_feature_columns = deep_feature_columns
        self.embedding_size = embedding_size
        self.hidden_units = hidden_units
        self.learning_rate = learning_rate
        self.output_dim = output_dim
        self.activate_fn = activate_fn
        self.log_loss = log_loss
        self.wide_optimizer = wide_optimizer
        self.deepfm_optimizer = deepfm_optimizer
        self.dropout_keep_prob_fm = 1 - dropout_fm
        self.dropout_keep_prob_deep = 1 - dropout_deep
        self.regression = regression
        self.share_embedding_dict = share_embedding_dict
        self.pre = None
        self.loss = None
        self.__check_fm_columns()
        if self.deepfm_optimizer is None:
            self.deepfm_optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate)
        if self.wide_optimizer is None:
            self.wide_optimizer = tf.train.FtrlOptimizer(learning_rate=self.learning_rate)
        self.embeddings = {}

    def get_optimizers(self):
        return [self.deepfm_optimizer, self.wide_optimizer]

    def set_optimizers(self, optimizers):
        self.deepfm_optimizer, self.wide_optimizer = optimizers

    def build_graph(self, features, targets, mode=None, config=None):
        num_ps_replicas = config.num_ps_replicas if config else 0
        partitioner = partitioned_variables.min_max_variable_partitioner(
            max_partitions=num_ps_replicas,
            min_slice_size=64 << 20)
        builder = _LazyBuilder(features)
        with tf.variable_scope(WIDE_SCOPE, reuse=tf.AUTO_REUSE, partitioner=partitioner):
            fm_first_output = tf.feature_column.linear_model(
                features, self.wide_feature_columns, units=self.output_dim)

        with tf.variable_scope(DEEPFM_SCOPE, reuse=tf.AUTO_REUSE, partitioner=partitioner):
            for k, deep_feature_column in enumerate(self.deep_feature_columns):
                if isinstance(deep_feature_column, _CategoricalColumn):
                    if self.share_embedding_dict is not None:
                        index = self.share_embedding_dict[k]
                    else:
                        index = "%d" % k
                    self.embeddings[k] = tf.get_variable(
                        name="feature_embedding_%s" % index,
                        shape=[deep_feature_column._num_buckets, self.embedding_size],
                        dtype=tf.float32
                    )

            fm_embedding_list = []
            fm_embedding_list_square = []
            deep_input_embedding = []
            deep_input_no_embedding = []
            for k, feature_column in enumerate(self.deep_feature_columns):
                if isinstance(feature_column, _CategoricalColumn):
                    sparse_tensors = feature_column._get_sparse_tensors(builder)
                    sparse_ids = sparse_tensors.id_tensor
                    sparse_weights = sparse_tensors.weight_tensor
                    feature_embedding = safe_embedding_lookup_sparse(
                        self.embeddings[k], sparse_ids, sparse_weights=sparse_weights, combiner="sum")
                    deep_input_embedding.append(feature_embedding)
                    if feature_column in self.fm_feature_columns:
                        fm_embedding_list.append(feature_embedding)
                        fm_feature_embedding_square = safe_embedding_lookup_sparse(
                            self.embeddings[k], sparse_ids, sparse_weights=sparse_weights, combiner="sum",
                            embedding_fun=math_ops.abs)
                        fm_embedding_list_square.append(fm_feature_embedding_square)
                else:
                    deep_input_no_embedding.append(feature_column)

            fm_second_input = DeepFM.tf_concat(fm_embedding_list, 1)
            feature_embeddings = tf.reshape(fm_second_input, (-1, len(self.embeddings), self.embedding_size))
            fm_second_input_square = DeepFM.tf_concat(fm_embedding_list_square, 1)
            feature_embeddings_square = tf.reshape(fm_second_input_square,
                                                   (-1, len(self.embeddings), self.embedding_size))
            summed_square_feature_embeddings = tf.square(tf.reduce_sum(feature_embeddings, 1))
            squared_sum_feature_embeddings = tf.reduce_sum(tf.square(feature_embeddings_square), 1)
            fm_second = 0.5 * tf.subtract(summed_square_feature_embeddings, squared_sum_feature_embeddings)
            if self.dropout_keep_prob_fm < 1.0:
                fm_second = tf.nn.dropout(fm_second, self.dropout_keep_prob_fm)
            fm_second_output = tf.layers.dense(fm_second, units=self.output_dim,
                                               kernel_initializer=tf.glorot_uniform_initializer())

            if len(deep_input_no_embedding) > 0:
                deep_input_embedding.append(tf.feature_column.input_layer(features, deep_input_no_embedding))
            deep_input = DeepFM.tf_concat(deep_input_embedding, 1)
            hidden_units = copy.deepcopy(self.hidden_units)
            hidden_units.append(self.output_dim)
            deep_output = self.__build_deep_layers(deep_input, hidden_units)
            if self.dropout_keep_prob_deep < 1.0:
                deep_output = tf.nn.dropout(deep_output, self.dropout_keep_prob_deep)

        output_tensor = tf.add(tf.add(fm_second_output, fm_first_output), deep_output)
        self.loss = None
        if self.regression:
            self.pre = output_tensor
        elif self.output_dim <= 2:
            self.pre = tf.nn.sigmoid(output_tensor)
        else:
            self.pre = tf.nn.softmax(output_tensor)

        def _train_op_fn(loss):
            train_ops = []
            global_step = training_util.get_global_step()
            train_ops.append(
                self.wide_optimizer.minimize(loss, global_step=global_step, var_list=ops.get_collection(
                    ops.GraphKeys.TRAINABLE_VARIABLES, scope=WIDE_SCOPE)))
            train_ops.append(
                self.deepfm_optimizer.minimize(loss, global_step=global_step, var_list=ops.get_collection(
                    ops.GraphKeys.TRAINABLE_VARIABLES, scope=DEEPFM_SCOPE)))
            train_op = control_flow_ops.group(*train_ops)
            with ops.control_dependencies([train_op]):
                with ops.colocate_with(global_step):
                    return state_ops.assign_add(global_step, 1)
        if targets is not None:
            if self.regression:
                self.loss = tf.losses.mean_squared_error(targets, output_tensor)
            elif self.output_dim <= 2:
                self.loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=targets, logits=output_tensor))
            else:
                self.loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=targets, logits=output_tensor))
            self.train_op = _train_op_fn(self.loss)

    def __check_fm_columns(self):
        if len(self.fm_feature_columns) < 2:
            raise ValueError("Fm Feature Columns Must Have At Least Two Elements")
        for f in self.fm_feature_columns:
            if f not in self.deep_feature_columns:
                raise ValueError("Fm Side Columns Must Be The Subset Of Deep Side")
            if not isinstance(f, _CategoricalColumn):
                raise ValueError("Fm Side Columns Must Be Categorical Columns")

    def __build_deep_layers(self, net, hidden_units):
        for num_hidden_units in hidden_units:
            net = tf.layers.dense(
                net,
                units=num_hidden_units,
                activation=self.activate_fn,
                kernel_initializer=tf.glorot_uniform_initializer())
            if self.dropout_keep_prob_deep < 1.0:
                net = tf.nn.dropout(net, self.dropout_keep_prob_deep)
        return net

    def get_metrics(self, targets, preds):
        metrics = {
            'label/mean': tf.metrics.mean(targets),
            'prediction/mean': tf.metrics.mean(preds),
            "mean_absolute_error": tf.metrics.mean_absolute_error(targets, preds),
            "mean_squared_error": tf.metrics.mean_squared_error(targets, preds),
        }
        if not self.regression:
            metrics.update(
                {
                    'accuracy': tf.metrics.accuracy(labels=targets,
                                                    predictions=tf.to_float(tf.greater_equal(preds, 0.5))),
                    'auc': tf.metrics.auc(targets, preds),
                }
            )
        return metrics

    def get_eval_summary(self, targets, preds):
        summary = {}
        for key, value in self.get_metrics(targets, preds).items():
            summary[key] = value[1]
        return summary

    @staticmethod
    def tf_concat(embeddings, index):
        x = None
        for i, one in enumerate(embeddings):
            if i == 0:
                x = one
            else:
                x = tf.concat([x, one], index)
        return x
