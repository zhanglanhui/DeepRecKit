#!/usr/bin/python
# -*- encoding: utf-8 -*-
# @Author: gavinlzhang
# @Date: 2022/08/08
import json
import os
import random
import glob
import gzip
import csv
import numpy as np
import tensorflow as tf
import pandas as pd

class FileNameIterator(tf.estimator.RunConfig):
    def __int__(self, config):
        super().__int__(config)

    def get_file_names(self, train_paths):
        all_files = []
        for path in train_paths:
            files = glob.glob(os.path.join(path, '**', '*.csv'), recursive=True)
            all_files.extend(files)
        return all_files

    def get_tfrecord_file_names(self, train_paths):
        all_files = []
        for path in train_paths:
            files = glob.glob(os.path.join(path, '**', '*.tfrecord.gz'), recursive=True)
            all_files.extend(files)
        return all_files

    def write_to_file(self, predict_path, results):
        with open(predict_path, 'w') as file:
            for result in results:
                file.write(str(result) + '\n')

    def write_to_file_list(self, predict_path, results):
        with open(predict_path, 'w') as file:
            for result in results:
                res = {}
                # Convert TensorFlow outputs into JSON-friendly Python values.
                for key, value in result.items():
                    if isinstance(value, np.ndarray):
                        res[key] = value.tolist()
                    elif isinstance(value, bytes):
                        res[key] = str(value.decode('utf-8'))
                    else:
                        res[key] = value
                file.write(str(res) + '\n')

    def write_to_file_list_append(self, predict_path, results):
        with open(predict_path, 'a') as file:
            file.write(str(results) + '\n')

class DataInput(object):
    def __init__(self, feature_conf: dict):
        self.feature_conf = feature_conf

    def convert_csv_to_tfrecord(self, csv_file, tfrecord_file):
        data = pd.read_csv(csv_file)
        def _bytes_feature(value):
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

        def _float_feature(value):
            return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

        def _int64_feature(value):
            return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

        def serialize_example(row):
            feature_description = {}
            for feat_name, feat_type in self.feature_conf["samples"]["item"].items():
                if feat_type["dtype"] == "int":
                    feature_description[feat_name] = _int64_feature(int(row[feat_name]))
                elif feat_type["dtype"] == "float":
                    feature_description[feat_name] = _float_feature(float(row[feat_name]))
                elif feat_type["dtype"] == "string":
                    feature_description[feat_name] = _bytes_feature(str(row[feat_name]).encode())
            for feat_name, feat_type in self.feature_conf["samples"]["user"].items():
                if feat_type["dtype"] == "int":
                    feature_description[feat_name] = _int64_feature(int(row[feat_name]))
                elif feat_type["dtype"] == "float":
                    feature_description[feat_name] = _float_feature(float(row[feat_name]))
                elif feat_type["dtype"] == "string":
                    feature_description[feat_name] = _bytes_feature(str(row[feat_name]).encode())

            example_proto = tf.train.Example(features=tf.train.Features(feature=feature_description))
            return example_proto.SerializeToString()

        with tf.io.TFRecordWriter(tfrecord_file) as writer:
            for index, row in data.iterrows():
                example = serialize_example(row)
                writer.write(example)

    def serialize_example(self, row):
        def _bytes_feature(value):
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

        def _float_feature(value):
            return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

        def _int64_feature(value):
            return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

        feature_description = {}
        for ctype in ("user", "item"):
            for feat_name, feat_type in self.feature_conf["samples"][ctype].items():
                if feat_type["dtype"] == "int":
                    feature_description[feat_name] = _int64_feature(int(row[feat_name]))
                elif feat_type["dtype"] == "float":
                    feature_description[feat_name] = _float_feature(float(row[feat_name]))
                elif feat_type["dtype"] == "string":
                    if "is_seq" in feat_type and feat_type["is_seq"]:
                        feature_description[feat_name] = _bytes_feature(
                            tf.io.serialize_tensor(row[feat_name].split(",")))
                    else:
                        feature_description[feat_name] = _bytes_feature(str(row[feat_name]).encode())
        example_proto = tf.train.Example(features=tf.train.Features(feature=feature_description))
        return example_proto.SerializeToString()

    def convert_dataset_to_tfrecord(self, dataset, tfrecord_file):
        options = tf.io.TFRecordOptions(compression_type='GZIP')
        with tf.io.TFRecordWriter(tfrecord_file, options=options) as writer:
            for index, row in dataset.iterrows():
                example = self.serialize_example(row)
                writer.write(example)

    def convert_dataset_to_tfrecord_json(self, dataset, tfrecord_file, exclude_field=None):
        options = tf.io.TFRecordOptions(compression_type='GZIP')
        with tf.io.TFRecordWriter(tfrecord_file, options=options) as writer:
            for index, row in dataset.iterrows():
                data = row[1]
                dicts = json.loads(data)
                if exclude_field: dicts = {k: v for k, v in dicts.items() if k not in exclude_field}
                example = self.serialize_example(dicts)
                writer.write(example)

    def get_dataset_from_tfrecord_pair(self, file_name, num_epochs, is_shuffle,
                                       batch_size, is_train, com, filter_fn=None):
        def _parse_function(example_proto):
            context_features = {}
            feat_names = []
            for feat_name, feat_type in self.feature_conf.items():
                if feat_type["dtype"] in ['tf.int32', 'tf.int', 'tf.int64']:
                    context_features[feat_name] = tf.FixedLenFeature([], tf.int32)
                    context_features["Neg" + feat_name] = tf.FixedLenFeature([], tf.int32)
                elif feat_type["dtype"] in ['tf.float32', 'tf.float']:
                    context_features[feat_name] = tf.FixedLenFeature([], tf.float32)
                    context_features["Neg" + feat_name] = tf.FixedLenFeature([], tf.float32)
                else:
                    if feat_type["ftype"] == "tf.VarLenFeature":
                        context_features[feat_name] = tf.VarLenFeature(tf.string)
                        context_features["Neg" + feat_name] = tf.VarLenFeature(tf.string)
                    else:
                        context_features[feat_name] = tf.FixedLenFeature([], tf.string)
                        context_features["Neg" + feat_name] = tf.FixedLenFeature([], tf.string)
                feat_names.append(feat_name)
                feat_names.append("Neg" + feat_name)
            # print(context_features)
            parsed_example = tf.io.parse_single_sequence_example(example_proto, context_features)
            pos_features = dict()
            neg_features = dict()
            for x in feat_names:
                if "Neg" in x:
                    neg_features[x[3:]] = parsed_example[0][x]
                else:
                    pos_features[x] = parsed_example[0][x]
            return pos_features, neg_features
        if is_train:
            option = tf.data.Options()
            option.experimental_deterministic = False
        else:
            option = tf.data.Options()
            option.experimental_deterministic = True
        dataset = tf.data.TFRecordDataset(file_name, compression_type=com)
        dataset = dataset.with_options(option).map(_parse_function)
        if filter_fn is not None:
            dataset = dataset.filter(filter_fn)
        # Shuffle, repeat, and batch the examples.
        if is_shuffle:
            dataset = dataset.shuffle(buffer_size=10000)
        dataset = dataset.repeat(num_epochs).batch(batch_size)
        return dataset

    def get_dataset_from_tfrecord(self, file_names, num_epochs, is_shuffle,
                                  batch_size, is_train, com, filter_fn=None):
        def _parse_function(example_proto):
            context_features = {}
            feat_names = []
            for feat_name, feat_type in self.feature_conf.items():
                if feat_type["dtype"] in ['tf.int32', 'tf.int', 'tf.int64']:
                    context_features[feat_name] = tf.io.FixedLenFeature([], tf.int64)
                elif feat_type["dtype"] in ['tf.float32', 'tf.float']:
                    context_features[feat_name] = tf.io.FixedLenFeature([], tf.float32)
                elif feat_type["dtype"] == "tf.string":
                    if feat_type["ftype"] == "tf.VarLenFeature":
                        context_features[feat_name] = tf.VarLenFeature(tf.string)
                    else:
                        context_features[feat_name] = tf.FixedLenFeature([], tf.string)
                feat_names.append(feat_name)
            parsed_example = tf.io.parse_single_sequence_example(example_proto, context_features)
            features = dict()
            target = dict()
            for x in feat_names:
                if "label" in x:
                    target[x] = parsed_example[0][x]
                else:
                    features[x] = parsed_example[0][x]
            return features, target
        option = tf.data.Options()
        option.experimental_deterministic = not is_train
        files_dataset = tf.data.Dataset.from_tensor_slices(file_names)
        dataset = files_dataset.interleave(
            lambda filename: tf.data.TFRecordDataset(filename, compression_type=com,
                                                     num_parallel_reads=tf.data.experimental.AUTOTUNE),
            cycle_length=tf.data.experimental.AUTOTUNE,
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        ).with_options(option)
        # dataset = dataset.with_options(option)
        # files = tf.data.Dataset.list_files([file_name])
        # dataset = files.apply(
        #     tf.contrib.data.parallel_interleave(tf.data.TFRecordDataset, cycle_length=FLAGS.num_parallel_readers))
        if filter_fn is not None:
            dataset = dataset.filter(filter_fn)
        # Shuffle, repeat, and batch the examples.
        if is_shuffle:
            dataset = dataset.shuffle(buffer_size=batch_size * 5)
        dataset = dataset.map(_parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.repeat(num_epochs).batch(batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return dataset

    def get_dataset_from_tfrecord_v2(self, file_names, num_epochs, is_shuffle,
                                     batch_size, is_train, com, filter_fn=None):
        def _parse_function(example_proto):
            context_features = {}
            feat_names = []
            for feat_name, feat_type in self.feature_conf.items():
                shape = int(feat_type["shape"])
                if feat_type["dtype"] in ['tf.int32', 'tf.int', 'tf.int64']:
                    context_features[feat_name] = tf.io.FixedLenFeature([shape], tf.int64)
                elif feat_type["dtype"] in ['tf.float32', 'tf.float']:
                    context_features[feat_name] = tf.io.FixedLenFeature([shape], tf.float32)
                elif feat_type["dtype"] == "tf.string":
                    if feat_type["ftype"] == "tf.VarLenFeature":
                        context_features[feat_name] = tf.VarLenFeature(tf.string)
                    else:
                        context_features[feat_name] = tf.FixedLenFeature([shape], tf.string)
                feat_names.append(feat_name)
            parsed_example = tf.io.parse_single_example(example_proto, features=context_features)
            features = dict()
            for name in feat_names:
                features[name] = parsed_example[name]
            target = features.pop("label")
            return features, target
        option = tf.data.Options()
        option.experimental_deterministic = not is_train
        files_dataset = tf.data.Dataset.from_tensor_slices(file_names)
        dataset = files_dataset.interleave(
            lambda filename: tf.data.TFRecordDataset(filename, compression_type=com,
                                                     num_parallel_reads=tf.data.experimental.AUTOTUNE),
            cycle_length=tf.data.experimental.AUTOTUNE,
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        ).with_options(option)
        # dataset = dataset.with_options(option)
        # files = tf.data.Dataset.list_files([file_name])
        # dataset = files.apply(
        #     tf.contrib.data.parallel_interleave(tf.data.TFRecordDataset, cycle_length=FLAGS.num_parallel_readers))
        if filter_fn is not None:
            dataset = dataset.filter(filter_fn)
        # Shuffle, repeat, and batch the examples.
        if is_shuffle:
            dataset = dataset.shuffle(buffer_size=batch_size * 5)
        dataset = dataset.map(_parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.repeat(num_epochs).batch(batch_size)
        dataset = dataset.prefetch(1)
        return dataset

    def get_dataset_from_tfrecord_v3(self, file_names, num_epochs, is_shuffle,
                                     batch_size, is_train, com, filter_fn=None):
        def _parse_function(example_proto):
            context_features = {}
            feat_names = []
            for feat_name, feat_type in self.feature_conf.items():
                shape = int(feat_type["shape"])
                if feat_type["dtype"] in ['tf.int32', 'tf.int', 'tf.int64']:
                    if feat_type["ftype"] == "tf.VarLenFeature":
                        context_features[feat_name] = tf.VarLenFeature(tf.int64)
                    else:
                        context_features[feat_name] = tf.FixedLenFeature([shape], tf.int64)
                elif feat_type["dtype"] in ['tf.float32', 'tf.float']:
                    if feat_type["ftype"] == "tf.VarLenFeature":
                        context_features[feat_name] = tf.VarLenFeature(tf.float32)
                    else:
                        context_features[feat_name] = tf.FixedLenFeature([shape], tf.float32)
                elif feat_type["dtype"] == "tf.string":
                    if feat_type["ftype"] == "tf.VarLenFeature":
                        context_features[feat_name] = tf.VarLenFeature(tf.string)
                    else:
                        context_features[feat_name] = tf.FixedLenFeature([shape], tf.string)
                # print(feat_name, context_features[feat_name])
                feat_names.append(feat_name)
            parsed_example = tf.io.parse_single_example(example_proto, features=context_features)
            features = dict()
            for name in feat_names:
                features[name] = parsed_example[name]
            target = features.pop("label")
            return features, target
        option = tf.data.Options()
        option.experimental_deterministic = not is_train
        files_dataset = tf.data.Dataset.from_tensor_slices(file_names)
        dataset = files_dataset.interleave(
            lambda filename: tf.data.TFRecordDataset(filename, compression_type=com,
                                                     num_parallel_reads=tf.data.experimental.AUTOTUNE),
            cycle_length=tf.data.experimental.AUTOTUNE,
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        ).with_options(option)
        if filter_fn is not None:
            dataset = dataset.filter(filter_fn)
        if is_shuffle:
            dataset = dataset.shuffle(buffer_size=batch_size * 5)
        dataset = dataset.map(_parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.repeat(num_epochs).batch(batch_size)
        dataset = dataset.prefetch(1)
        return dataset

    def get_dataset_from_csv_v3(self, file_names, num_epochs, is_shuffle,
                                batch_size, is_train, com=None, filter_fn=None, field_delim=","):
        if not file_names:
            raise ValueError("file_names is empty")

        with open(file_names[0], "r", encoding="utf-8") as csv_file:
            reader = csv.reader(csv_file, delimiter=field_delim)
            header = next(reader)

        if "label" not in header:
            raise ValueError("CSV header must contain label column")

        missing_columns = [name for name in self.feature_conf.keys() if name not in header]
        if missing_columns:
            raise ValueError("CSV header missing columns: {}".format(",".join(missing_columns)))

        def _is_int_dtype(dtype):
            return dtype in ['tf.int32', 'tf.int', 'tf.int64', 'int', tf.int32, tf.int64]

        def _is_float_dtype(dtype):
            return dtype in ['tf.float32', 'tf.float', 'float', 'float32', tf.float32]

        def _is_string_dtype(dtype):
            return dtype in ['tf.string', 'string', tf.string]

        def _is_var_len_feature(ftype):
            return ftype in ['tf.VarLenFeature', tf.VarLenFeature]

        def _get_scalar_default(feat_type):
            default_value = feat_type.get("default_value", "")
            if _is_int_dtype(feat_type["dtype"]):
                return tf.constant([int(default_value)], dtype=tf.int64)
            if _is_float_dtype(feat_type["dtype"]):
                return tf.constant([float(default_value)], dtype=tf.float32)
            return tf.constant([str(default_value)], dtype=tf.string)

        selected_names = [name for name in header if name in self.feature_conf]
        selected_indices = [header.index(name) for name in selected_names]
        record_defaults = []
        for name in selected_names:
            feat_type = self.feature_conf[name]
            shape = int(feat_type.get("shape", 1))
            if _is_var_len_feature(feat_type.get("ftype")) or shape > 1:
                record_defaults.append(tf.constant([""], dtype=tf.string))
            else:
                record_defaults.append(_get_scalar_default(feat_type))

        def _cast_tokens(values, feat_type):
            if _is_int_dtype(feat_type["dtype"]):
                return tf.strings.to_number(values, out_type=tf.int64)
            if _is_float_dtype(feat_type["dtype"]):
                return tf.strings.to_number(values, out_type=tf.float32)
            if _is_string_dtype(feat_type["dtype"]):
                return values
            return values

        def _get_sequence_default(feat_type, shape):
            default_value = feat_type.get("default_value", "")
            if isinstance(default_value, list):
                default_tokens = [str(x) for x in default_value]
            elif default_value in [None, ""]:
                default_tokens = []
            else:
                default_tokens = str(default_value).split(",")
            if shape > 1 and len(default_tokens) == 1:
                default_tokens = default_tokens * shape
            return _cast_tokens(tf.constant(default_tokens, dtype=tf.string), feat_type)

        def _parse_sequence_value(raw_value, feat_type, shape):
            default_tensor = _get_sequence_default(feat_type, shape)
            return tf.cond(
                tf.equal(tf.strings.length(raw_value), 0),
                lambda: default_tensor,
                lambda: _cast_tokens(tf.strings.split([raw_value], sep=",").values, feat_type)
            )

        def _parse_csv_line(line):
            parsed_columns = tf.io.decode_csv(
                line,
                record_defaults=record_defaults,
                field_delim=field_delim,
                use_quote_delim=True,
                select_cols=selected_indices
            )

            parsed_dict = dict(zip(selected_names, parsed_columns))
            features = {}
            for name in selected_names:
                feat_type = self.feature_conf[name]
                shape = int(feat_type.get("shape", 1))
                value = parsed_dict[name]
                if _is_var_len_feature(feat_type.get("ftype")):
                    parsed_value = _parse_sequence_value(value, feat_type, shape)
                    features[name] = tf.SparseTensor(
                        indices=tf.expand_dims(tf.range(tf.size(parsed_value), dtype=tf.int64), 1),
                        values=parsed_value,
                        dense_shape=tf.cast([tf.size(parsed_value)], tf.int64)
                    )
                elif shape > 1:
                    parsed_value = _parse_sequence_value(value, feat_type, shape)
                    features[name] = tf.reshape(parsed_value, [shape])
                elif _is_int_dtype(feat_type["dtype"]):
                    features[name] = tf.reshape(tf.cast(value, tf.int64), [shape])
                elif _is_float_dtype(feat_type["dtype"]):
                    features[name] = tf.reshape(tf.cast(value, tf.float32), [shape])
                else:
                    features[name] = tf.reshape(value, [shape])

            target = features.pop("label")
            return features, target

        option = tf.data.Options()
        option.experimental_deterministic = not is_train
        files_tensor = tf.convert_to_tensor(file_names, dtype=tf.string)
        files_dataset = tf.data.Dataset.from_tensor_slices(files_tensor)
        dataset = files_dataset.flat_map(
            lambda filename: tf.data.TextLineDataset(filename).skip(1)
        ).with_options(option)
        if is_shuffle:
            dataset = dataset.shuffle(buffer_size=batch_size * 5)
        dataset = dataset.map(_parse_csv_line, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        if filter_fn is not None:
            dataset = dataset.filter(filter_fn)
        dataset = dataset.repeat(num_epochs).batch(batch_size)
        dataset = dataset.prefetch(1)
        return dataset

    def serving_input_fn_from_feed(self):
        pass

    def serving_input_fn_from_string(self):
        # Build placeholders from the generated feature schema for model export.
        features = {}
        for feature_name, feature_info in self.feature_conf.items():
            if "label" in feature_name:
                continue
            dtype = feature_info['dtype']
            shape = int(feature_info['shape'])
            if dtype == "tf.int32":
                if shape > 1:
                    features[feature_name] = tf.placeholder(tf.int32, shape=[None, None], name=feature_name)
                else:
                    features[feature_name] = tf.placeholder(tf.int32, shape=[None, 1], name=feature_name)
            elif dtype == "tf.float32":
                if shape > 1:
                    features[feature_name] = tf.placeholder(tf.float32, shape=[None, None], name=feature_name)
                else:
                    features[feature_name] = tf.placeholder(tf.float32, shape=[None, 1], name=feature_name)
            elif dtype == 'tf.string':
                if shape > 1:
                    features[feature_name] = tf.placeholder(tf.string, shape=[None, None], name=feature_name)
                else:
                    features[feature_name] = tf.placeholder(tf.string, shape=[None, 1], name=feature_name)
        return tf.estimator.export.ServingInputReceiver(features, features)

class FeatureTransformer(object):
    def __init__(self, transformer_conf):
        # super(FeatureTransformer, self).__init__(
        #     transformer_conf=transformer_conf)
        self.transformer_conf = transformer_conf

    def get_model_features_groups(self):
        feature_column_dict = {}
        # clust_feats = {}
        for config in self.transformer_conf["feature_column_config_list"]:
            if isinstance(config["input_feature_name"], list):
                input_features = []
                for feature_name in config["input_feature_name"]:
                    if feature_name in feature_column_dict:
                        input_features.append(feature_column_dict[feature_name])
                    else:
                        # raise ValueError("Input feature name:{} not found in dictionary.".format(feature_name))
                        input_features.append(feature_name)
                if isinstance(config["output_feature_name"], list):
                    for id, y in enumerate(config["output_feature_name"]):
                        # print(input_features)
                        # print(config["ftype"](input_features,
                        #                       **config["parameters"]))
                        feature_column_dict[y] = config["ftype"](input_features,
                                                                 **config["parameters"])[id]
                        # clust_feats[y] = set(config["output_feature_name"])
                else:
                    feature_column_dict[config["output_feature_name"]] = config["ftype"](input_features,
                                                                                         **config["parameters"])
            elif isinstance(config["input_feature_name"], str):
                if config["input_feature_name"] not in feature_column_dict:
                    feature_column_dict[config["output_feature_name"]] = config["ftype"](
                        f'{config["input_feature_name"]}',
                        **config["parameters"])
                else:
                    # print(config["output_feature_name"])
                    # print(config["ftype"])
                    feature_column_dict[config["output_feature_name"]] = config["ftype"](
                        feature_column_dict[config["input_feature_name"]],
                        **config["parameters"])
            else:
                raise ValueError("Input feature name:{} not found in dictionary.".format(config["input_feature_name"]))
        # print(feature_column_dict["i_banner_id_category_embedding"])
        # print(feature_column_dict["u_banner_id_profile_category_embedding"])
        # print(feature_column_dict)
        feature_columns = {}
        group_config = self.transformer_conf["feature_column_group"]
        for group_key, group_value in group_config.items():
            group_columns = []
            # black_list = set()
            for feature_name in group_value:
                if feature_name in feature_column_dict:
                    group_columns.append(feature_column_dict[feature_name])
                    # if isinstance(feature_column_dict[feature_name], list):
                    #     print("feature_name", feature_name)
                    #     print("feature_column_dict[feature_name]", feature_column_dict[feature_name])
                    #     raise ValueError("Input feature name:{} not found in dictionary.".format("x"))
                    # if feature_name in clust_feats:
                    #     black_list.update(clust_feats[feature_name])
                # else:
                #     print("safdsd", feature_name)
            feature_columns[group_key] = group_columns
        return feature_columns
