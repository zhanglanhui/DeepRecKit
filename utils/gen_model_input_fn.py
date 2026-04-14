#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
Created on MAY 21, 2024
@author: zhanglanhui
"""
import sys
import json

if len(sys.argv) < 3:
    print("Please provide the path to the JSON file as a command line argument.")
    sys.exit(1)

json_file_path = sys.argv[1]
cross_file_path = sys.argv[2]

with open(json_file_path, 'r', encoding='utf-8') as json_file:
    data = json.load(json_file)

with open(cross_file_path, 'r', encoding='utf-8') as json_file1:
    data_cross = json.load(json_file1)

all_feats = data

type_dict = {
    "string": "tf.string",
    "float": "tf.float32",
    "int": "tf.int64",
}

transformer_config = {
    "features_groups": {
        "dense": []
        ,
        "sparse": []
        ,
        "weight": []
        ,
        "cross": []
        ,
    }
}

emb_feats = {}
numeric_list = []
emb_list = []
seq_list = []
indicator_list = []
share_emb_list = []
cross_emb_list = []
cator_list = []
with open("model_input_fn.py", "w") as file:
    file.write("import tensorflow as tf\n\n")
    # category
    # categorical_column_with_hash_bucket feat_name,num_buckets default_value=0
    # categorical_column_with_vocabulary_list feat_name,vocabulary_list=[] default_value=0
    file.write("# numeric_column and categorical_column\n")
    for feat_name, conf in all_feats.items():
        if conf["is_label"] or conf['dtype'] == "float":
            feature_column_code = f"{feat_name}_fc = tf.feature_column.numeric_column('{feat_name}', dtype={type_dict[conf['dtype']]})"
            if not conf["is_label"]: numeric_list.append(f"{feat_name}_fc")
        else:
            if "is_seq" not in conf or not conf["is_seq"]:
                feature_column_code = f"{feat_name}_fc = tf.feature_column.categorical_column_with_hash_bucket('{feat_name}', {conf['hash_size']}, dtype={type_dict[conf['dtype']]})"
                cator_list.append(f"{feat_name}_fc")
                emb_feats[f"{feat_name}_fc"] = conf['emb_dim']
            else:
                continue
        file.write(feature_column_code + "\n")
    file.write("\n")

    file.write("# seq_column\n")
    for feat_name, conf in all_feats.items():
        if "is_seq" in conf and conf["is_seq"]:
            feature_column_code = f"{feat_name}_seq_fc = tf.feature_column.sequence_categorical_column_with_hash_bucket('{feat_name}', {conf['hash_size']}, dtype={type_dict[conf['dtype']]})"
            file.write(feature_column_code + "\n")
            seq_list.append(f"{feat_name}_seq_fc")
            emb_feats[f"{feat_name}_seq_fc"] = conf['emb_dim']

    file.write("\n\n")
    file.write(f"feature_columns = []" + "\n")
    file.write(f"feature_columns += [" + ", ".join(
        [x for x in cator_list] + [x for x in seq_list]) + "]\n")

    file.write("\n\n")
    file.write("# indicator\n")
    for feat_name in cator_list:
        feature_column_code = f"{feat_name}_indicator = tf.feature_column.indicator_column({feat_name})"
        file.write(feature_column_code + "\n")
        indicator_list.append(f"{feat_name}_indicator")

    file.write("\n\n")
    file.write("# embedding_columns\n")
    # uin_embedding = tf.feature_column.embedding_column(uin, dimension=128)
    for f in cator_list:
        feature_column_code = f"{f}_embedding = tf.feature_column.embedding_column({f}, dimension={emb_feats[f]})"
        file.write(feature_column_code + "\n")
        emb_list.append(f"{f}_embedding")
    for f in seq_list:
        feature_column_code = f"{f}_embedding = tf.feature_column.embedding_column({f}, dimension={emb_feats[f]})"
        file.write(feature_column_code + "\n")
        emb_list.append(f"{f}_embedding")

    # shared_embedding_columns
    # print(emb_feats)
    file.write("\n\n")
    file.write("# shared_embedding_columns\n")
    for f in all_feats.keys():
        if "label" in f:
            continue
        if all_feats[f]["share_emb"] != "" and all_feats[f]["share_emb"] in all_feats.keys():
            tt = all_feats[f]["share_emb"]
            dim = emb_feats[f + "_fc"]
            feature_column_code = f"{f}_share_embedding = tf.feature_column.shared_embedding_columns([{f}_fc,{tt}_fc], dimension={dim})"
            file.write(feature_column_code + "\n")
            share_emb_list.append(f"{f}_share_embedding")
    # print(share_emb_list)

    # cross
    # cross_columns = [tf.feature_column.crossed_column([age_bucket,gain_bucket],hash_bucket_size=36),
    # tf.feature_column.crossed_column([gain_bucket,loss_bucket],hash_bucket_size=16)]
    for key, vals in data_cross.items():
        # print("xx")
        # print(key, vals)
        # sex_department = tf.feature_column.crossed_column([department, sex], 10)
        # sex_department = tf.feature_column.indicator_column(sex_department)
        names = vals["names"]
        hash_bucket_size = vals["hash_bucket_size"]
        emb_size = vals["emb_size"]
        feature_column_code = f"{key} = tf.feature_column.crossed_column({names}, hash_bucket_size={hash_bucket_size})"
        file.write(feature_column_code + "\n")
        feature_column_code = f"{key}_indicator = tf.feature_column.indicator_column({key})"
        file.write(feature_column_code + "\n")
        indicator_list.append(f"{key}_indicator")
        feature_column_code = f"{key}_embedding = tf.feature_column.embedding_column({key}_indicator, dimension={emb_size})"
        file.write(feature_column_code + "\n")
        emb_list.append(f"{key}_embedding")

    file.write("\n\n")
    # with open("model_json_conf.py", "a") as file:
    file.write("FEATURE_TRANSFORM_CONFIG = ")
    json.dump(transformer_config, file, indent=4)
    file.write("\n\n")
    # FEATURE_TRANSFORM_CONFIG["features_groups"]["dense"].append(uin_embedding)
    for feat_name in indicator_list:
        # feature_column_code = f"FEATURE_TRANSFORM_CONFIG['features_groups']['dense'].append({feat_name})"
        # file.write(feature_column_code + "\n")
        feature_column_code = f"FEATURE_TRANSFORM_CONFIG['features_groups']['sparse'].append({feat_name})"
        file.write(feature_column_code + "\n")
    for feat_name in emb_list:
        # print(feat_name)
        feature_column_code = f"FEATURE_TRANSFORM_CONFIG['features_groups']['dense'].append({feat_name})"
        file.write(feature_column_code + "\n")
        # feature_column_code = f"FEATURE_TRANSFORM_CONFIG['features_groups']['sparse'].append({feat_name})"
        # file.write(feature_column_code + "\n")
    for feat_name in numeric_list:
        feature_column_code = f"FEATURE_TRANSFORM_CONFIG['features_groups']['dense'].append({feat_name})"
        file.write(feature_column_code + "\n")
        feature_column_code = f"FEATURE_TRANSFORM_CONFIG['features_groups']['sparse'].append({feat_name})"
        file.write(feature_column_code + "\n")
    for feat_name in share_emb_list:
        feature_column_code = f"FEATURE_TRANSFORM_CONFIG['features_groups']['dense'].append({feat_name})"
        file.write(feature_column_code + "\n")
        # feature_column_code = f"FEATURE_TRANSFORM_CONFIG['features_groups']['sparse'].append({feat_name})"
        # file.write(feature_column_code + "\n")
    # {feat_name: {"dtype": dtype_dict[category_feats[feat_name][1]]}}
    # with open("model_json_conf.py", "w") as config_file:

    # FEATURE_TRANSFORM_CONFIG["features_groups"]["dense"].append(uin_embedding)

