# -*- encoding:utf-8 -*-

import tensorflow as tf
import tensorflow.keras as K

class NNLayers:
    @staticmethod
    def build_deep_layers(net, hidden_units, activate_fn, dropout=0.0, name="dnn"):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            for idx, num_hidden_units in enumerate(hidden_units):
                net = tf.layers.dense(
                    net,
                    units=num_hidden_units,
                    activation=activate_fn,
                    kernel_initializer=tf.glorot_uniform_initializer(),
                    name=name + str(idx))
                if 1.0 > dropout > 0.0:
                    net = tf.nn.dropout(net, keep_prob=1 - dropout)
            return net

    @staticmethod
    def build_deep_layers_bn(net, hidden_units, activation, training, name="dnn"):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            for idx, num_hidden_units in enumerate(hidden_units):
                net = K.layers.Dense(
                    units=num_hidden_units,
                    activation=activation,
                    name=name + str(idx))(net)
                net = K.layers.BatchNormalization(
                    name="{0}_batch_normalization_{1}".format(name, idx))(net, training=training)
                # if 1.0 > dropout > 0.0:
                #     net = tf.nn.dropout(net, keep_prob=1 - dropout)
            return net

    @staticmethod
    def build_deep_layers_fei(net, hidden_units, activate_fn, dropout=0.0, name="dnn_fei"):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            nets = []
            net = tf.nn.dropout(net, keep_prob=1 - dropout)
            nets.append(net)
            for idx, num_hidden_units in enumerate(hidden_units):
                net = tf.layers.dense(
                    net,
                    units=num_hidden_units,
                    activation=activate_fn,
                    kernel_initializer=tf.glorot_uniform_initializer(),
                    name=name + str(idx))
                nets.append(net)
            return nets

    @staticmethod
    def build_deep_layers_freeze(net, hidden_units, activate_fn, dropout=0.0, name="dnn"):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            for idx, num_hidden_units in enumerate(hidden_units):
                net = tf.layers.dense(
                    net,
                    units=num_hidden_units,
                    activation=activate_fn,
                    kernel_initializer=tf.glorot_uniform_initializer(),
                    name=name + str(idx),
                    trainable=False)
                if 1.0 > dropout > 0.0:
                    net = tf.nn.dropout(net, keep_prob=1 - dropout)
            return net

    @staticmethod
    def get_cross_layer(input_tensor, cross_layers, cross_dropout=0.0, name="cross"):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            dims = input_tensor.shape[1]
            origin_tensor = input_tensor
            output_tensor = origin_tensor
            for i in range(cross_layers):
                w = tf.get_variable(name=name + "_cross_w_%d" % i, shape=[dims], dtype=tf.float32,
                                    initializer=tf.random_normal_initializer(mean=0.0, stddev=1.0, seed=None,
                                                                             dtype=tf.float32))
                b = tf.get_variable(name=name + "_cross_b_%d" % i, shape=[dims], dtype=tf.float32,
                                    initializer=tf.random_normal_initializer(mean=0.0, stddev=1.0, seed=None,
                                                                             dtype=tf.float32))
                trans = tf.reshape(output_tensor, [-1, 1, dims])
                dot = tf.tensordot(trans, w, axes=1)
                output_tensor = origin_tensor * dot + b + output_tensor
                if cross_dropout is not None and 1.0 > cross_dropout > 0.0:
                    output_tensor = tf.nn.dropout(output_tensor, keep_prob=1 - cross_dropout)
            return output_tensor

    @staticmethod
    def get_cross_matrix_layer(input_tensor, cross_layers, cross_dropout=0.0, name="cross"):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            dims = input_tensor.shape[1]
            origin_tensor = input_tensor
            output_tensor = origin_tensor
            for i in range(cross_layers):
                w = tf.get_variable(name=name + "_cross_w_%d" % i, shape=[dims, dims], dtype=tf.float32,
                                    initializer=tf.random_normal_initializer(mean=0.0, stddev=1.0, seed=None,
                                                                             dtype=tf.float32))
                b = tf.get_variable(name=name + "_cross_b_%d" % i, shape=[dims], dtype=tf.float32,
                                    initializer=tf.random_normal_initializer(mean=0.0, stddev=1.0, seed=None,
                                                                             dtype=tf.float32))
                dot = tf.matmul(output_tensor, w)
                output_tensor = origin_tensor * (dot + b) + output_tensor
                if cross_dropout is not None and 1.0 > cross_dropout > 0.0:
                    output_tensor = tf.nn.dropout(output_tensor, keep_prob=1 - cross_dropout)
            return output_tensor

    @staticmethod
    def get_new_cross_layer(input_tensor, cross_layers, cross_dropout=0.0, name="cross"):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            dims = input_tensor.shape[1]
            origin_tensor = input_tensor
            output_tensor = origin_tensor
            for i in range(cross_layers):
                old_tensor = output_tensor
                w = tf.get_variable(name=name + "_cross_w_%d" % i, shape=[dims], dtype=tf.float32,
                                    initializer=tf.random_normal_initializer(mean=0.0, stddev=1.0, seed=None,
                                                                             dtype=tf.float32))
                b = tf.get_variable(name=name + "_cross_b_%d" % i, shape=[dims], dtype=tf.float32,
                                    initializer=tf.random_normal_initializer(mean=0.0, stddev=1.0, seed=None,
                                                                             dtype=tf.float32))
                trans = tf.reshape(output_tensor, [-1, 1, dims])
                dot = tf.tensordot(trans, w, axes=1)
                output_tensor = origin_tensor * dot + b + output_tensor
                if cross_dropout is not None and 1.0 > cross_dropout > 0.0:
                    output_tensor = tf.nn.dropout(output_tensor, keep_prob=1 - cross_dropout)
                output_tensor = tf.concat([output_tensor, old_tensor], axis=-1)
            return output_tensor

    @staticmethod
    def get_compressed_layer(input_tensor, field_nums, compressed_layers, embedding_size,
                             compressed_l2=0.0, compressed_file_split_half=True, name="compressed"):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            field_nums = [field_nums]
            filters = []
            bias = []
            for i, size in enumerate(compressed_layers):
                filters.append(tf.get_variable(name='filter' + str(i), shape=[1, field_nums[-1] * field_nums[0], size],
                                               dtype=tf.float32,
                                               initializer=tf.initializers.glorot_uniform(seed=1024 + i),
                                               regularizer=tf.keras.regularizers.l2(compressed_l2)))
                bias.append(tf.get_variable(name='bias' + str(i), shape=[size], dtype=tf.float32,
                                            initializer=tf.keras.initializers.Zeros(),
                                            regularizer=tf.keras.regularizers.l2(compressed_l2)))
                if compressed_file_split_half:
                    if i != len(compressed_layers) - 1 and size % 2 > 0:
                        raise ValueError(
                            "layer_size must be even number except for the last layer when split_half=True")
                    field_nums.append(size // 2)
                else:
                    field_nums.append(size)
            final_result = []
            hidden_nn_layers = [input_tensor]
            dim = embedding_size
            split_tensor0 = tf.split(hidden_nn_layers[0], dim * [1], 2)
            for idx, layer_size in enumerate(compressed_layers):
                split_tensor = tf.split(hidden_nn_layers[-1], dim * [1], 2)
                dot_result_m = tf.matmul(split_tensor0, split_tensor, transpose_b=True)
                dot_result_o = tf.reshape(dot_result_m, shape=[dim, -1, field_nums[0] * field_nums[idx]])
                dot_result = tf.transpose(dot_result_o, perm=[1, 0, 2])
                curr_out = tf.nn.conv1d(dot_result, filters=filters[idx], stride=1, padding='VALID')
                curr_out = tf.nn.bias_add(curr_out, bias[idx])
                curr_out = tf.nn.relu(curr_out)
                curr_out = tf.transpose(curr_out, perm=[0, 2, 1])
                if compressed_file_split_half:
                    if idx != len(compressed_layers) - 1:
                        next_hidden, direct_connect = tf.split(curr_out, 2 * [layer_size // 2], 1)
                    else:
                        direct_connect = curr_out
                        next_hidden = 0
                else:
                    direct_connect = curr_out
                    next_hidden = curr_out
                final_result.append(direct_connect)
                hidden_nn_layers.append(next_hidden)
            result = tf.concat(final_result, axis=1)
            result = tf.reduce_sum(result, -1, keep_dims=False)
            return result

    @staticmethod
    def get_fm_layer(input_tensor, embedding_size, name="fm"):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            fm_feature_columns_size = int(input_tensor.get_shape()[-1].value / embedding_size)
            feature_embeddings = tf.reshape(input_tensor, (-1, embedding_size, fm_feature_columns_size))
            # summed_square_feature_embeddings = tf.square(tf.reduce_sum(feature_embeddings, 1))
            # squared_sum_feature_embeddings = tf.reduce_sum(tf.square(feature_embeddings), 1)
            # fm_second = 0.5 * tf.subtract(summed_square_feature_embeddings, squared_sum_feature_embeddings)
            square_of_sum = tf.square(tf.reduce_sum(
                feature_embeddings, axis=1, keepdims=True))
            sum_of_square = tf.reduce_sum(
                feature_embeddings * feature_embeddings, axis=1, keepdims=True)
            cross_term = square_of_sum - sum_of_square
            fm_second = 0.5 * tf.reduce_sum(cross_term, axis=2, keepdims=False)
            return fm_second

    @staticmethod
    def get_attention_layer(queries, keys, keys_length, attention_layers, name="attention"):
        '''
        queries:     [B, H]
        keys:        [B, T, H]
        keys_length: [B]
        '''
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            queries_hidden_units = queries.get_shape().as_list()[-1]
            queries = tf.tile(queries, [1, tf.shape(keys)[1]])
            queries = tf.reshape(queries, [-1, tf.shape(keys)[1], queries_hidden_units])
            din_all = tf.concat([queries, keys, queries - keys, queries * keys], axis=-1)
            d_layer_all = din_all
            for i, x in enumerate(attention_layers):
                d_layer_all = tf.layers.dense(d_layer_all, x, activation=tf.nn.relu, name='f1_att' + str(i),
                                              reuse=tf.AUTO_REUSE)
            d_layer_3_all = tf.layers.dense(d_layer_all, 1, activation=None, name='f3_att', reuse=tf.AUTO_REUSE)
            d_layer_3_all = tf.reshape(d_layer_3_all, [-1, 1, tf.shape(keys)[1]])
            outputs = d_layer_3_all
            # Mask
            keys_length = tf.reshape(keys_length, [-1, ])
            key_masks = tf.sequence_mask(keys_length, tf.shape(keys)[1])  # [B, T]
            key_masks = tf.expand_dims(key_masks, 1)  # [B, 1, T]
            paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
            outputs = tf.where(key_masks, outputs, paddings)  # [B, 1, T]
            # Scale
            outputs = outputs / (keys.get_shape().as_list()[-1] ** 0.5)
            # Activation
            outputs = tf.nn.softmax(outputs)  # [B, 1, T]
            # Weighted sum
            outputs = tf.matmul(outputs, keys)  # [B, 1, H]
            return tf.reshape(outputs, [-1, outputs.get_shape()[-1]])

    @staticmethod
    def get_cosine_layer(x, y, name="cosine", inner_product=False):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            dot = tf.reduce_sum(tf.multiply(x, y), axis=-1, keep_dims=True)
            if inner_product:
                return dot
            else:
                x_square = tf.reduce_sum(tf.square(x), axis=-1, keep_dims=True) + 0.00000001
                y_square = tf.reduce_sum(tf.square(y), axis=-1, keep_dims=True) + 0.00000001
                return dot / tf.sqrt(x_square * y_square)

    @staticmethod
    def get_origin_attention_layer(query, keys, keys_length, hidden, length, name):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            embedding_size = query.get_shape().as_list()[-1]
            query = tf.tile(query, [1, length])
            query = tf.reshape(query, [-1, length, embedding_size])
            din_all = tf.concat([query, keys, query - keys, query * keys], axis=-1)
            d_layer_all = din_all
            for i, x in enumerate(hidden):
                d_layer_all = tf.layers.dense(d_layer_all, x, activation=tf.nn.relu, name=name + 'f1_att' + str(i),
                                              reuse=tf.AUTO_REUSE)
            d_layer_3_all = tf.layers.dense(d_layer_all, 1, activation=None, name=name + 'f3_att', reuse=tf.AUTO_REUSE)
            d_layer_3_all = tf.reshape(d_layer_3_all, [-1, 1, tf.shape(keys)[1]])
            outputs = d_layer_3_all
            keys_length = tf.reshape(keys_length, [-1, ])
            key_masks = tf.sequence_mask(keys_length, tf.shape(keys)[1])  # [B, T]
            key_masks = tf.expand_dims(key_masks, 1)  # [B, 1, T]
            paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
            outputs = tf.where(key_masks, outputs, paddings)  # [B, 1, T]
            outputs = outputs / (keys.get_shape().as_list()[-1] ** 0.5)
            outputs = tf.nn.softmax(outputs)  # [B, 1, T]
            outputs = tf.matmul(outputs, keys)  # [B, 1, H]
            return tf.reshape(outputs, [-1, outputs.get_shape()[-1]])

    @staticmethod
    def get_custom_attention_layer(sequence_list, features, hidden=[64], name="custom_attention"):
        if len(sequence_list) < 2:
            raise Exception("Sequence List Len Must >= 2")
        keys = tf.feature_column.input_layer(features, sequence_list[0:-1])
        query = tf.feature_column.input_layer(features, sequence_list[-1:])
        length = len(sequence_list) - 1
        keys = tf.reshape(keys, [-1, length, query.get_shape()[-1]])
        keys_length = tf.count_nonzero(tf.reduce_sum(keys, axis=2),
                                       axis=1)
        attention_output = NNLayers.get_origin_attention_layer(query, keys, keys_length,
                                                               hidden, length, name)
        return attention_output
