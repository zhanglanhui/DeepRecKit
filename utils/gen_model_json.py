#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
Created on MAY 21, 2024
@author: zhanglanhui
"""
import sys
import json

if len(sys.argv) < 2:
    print("Please provide the path to the JSON file as a command line argument.")
    sys.exit(1)

json_file_path = sys.argv[1]

with open(json_file_path, 'r', encoding='utf-8') as json_file:
    data = json.load(json_file)

category_feats = data

dtype_dict = {
    'tf.int64': "int",
    'tf.string': "string",
    'tf.float32': "float"
}

sample_config = {
    "samples": {
        "context": {},
        "cross": {},
        "item": {},
        "user": {}
    }
}

with open("model_json_conf.py", "w") as file:
    file.write("FEATURE_CONFIG = ")
    for feat_name in category_feats.keys():
        if feat_name.startswith("i_") or feat_name.startswith("ic_"):
            sample_config["samples"]["item"].update({feat_name: {"dtype": category_feats[feat_name]["dtype"]}})
        else:
            sample_config["samples"]["user"].update({feat_name: {"dtype": category_feats[feat_name]["dtype"]}})
    # with open("model_json_conf.py", "w") as config_file:
    json.dump(sample_config, file, indent=4)

    # file.write("\n\n")
    # # with open("model_json_conf.py", "a") as file:
    # file.write("FEATURE_TRANSFORM_CONFIG = ")
    # for feat_name in category_feats.keys():
    #     transformer_config["features_groups"]["dense"].append(feat_name)
    #     transformer_config["features_groups"]["sparse"].append(feat_name)
    #     # {feat_name: {"dtype": dtype_dict[category_feats[feat_name][1]]}}
    # # with open("model_json_conf.py", "w") as config_file:
    # json.dump(transformer_config, file, indent=4)


