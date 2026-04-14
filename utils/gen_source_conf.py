#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
Created on MAY 21, 2024
@author: zhanglanhui
"""
import sys, os
import json
from collections import OrderedDict
if len(sys.argv) < 3:
    print("Please provide the path to the JSON file as a command line argument.")
    sys.exit(1)

fields_file = sys.argv[1]
json_file_dd = sys.argv[2]

if os.path.exists(json_file_dd):
    with open(json_file_dd, 'r', encoding='utf-8') as old_file:
        old_data = json.load(old_file)
else:
    old_data = {}

# print(json_file_dd)
# print(os.path.exists(json_file_dd))
# print(old_data)

with open(fields_file, 'r') as file:
    field_names_all = [line.strip() for line in file if line.strip()]
    field_names_valid = []
    for x in field_names_all:
        if "$" not in x:
            if ":" in x:
                field_names_valid.append(x.split(":")[0])
            else:
                field_names_valid.append(x)
    field_names_add = [x for x in field_names_all if x not in old_data and "$" not in x]
    field_names_del = [x for x in field_names_all if "$" in x]

print("not in json fields:", field_names_add)

default_params = {
    "dtype": "string",
    "default_value": "0",
    "zscore": False,
    "one_hot": False,
    "vocab_size": 300000,
    "is_seq": False,
    "emb_dim": 8,
    "embedding_name": "",
}

# "dtype": "int64",
# "default_value": 0,
# "zscore": false,
# "one_hot": false,
# "vocab_size": 300000

def sort_json_by_order(json_data, field_order):
    sorted_json_data = {field: json_data[field] for field in field_order}
    return sorted_json_data

json_data = {}
for field_name in field_names_add:
    tag = ""
    if ":" in field_name:
        tag = field_name.split(":")[1]
        field_name = field_name.split(":")[0]
    if field_name in old_data:
        continue
    json_data[field_name] = default_params.copy()
    if "label" in field_name:
        del json_data[field_name]["emb_dim"]
        del json_data[field_name]["embedding_name"]
        del json_data[field_name]["is_seq"]
        del json_data[field_name]["default_value"]
        del json_data[field_name]["zscore"]
        del json_data[field_name]["vocab_size"]
        del json_data[field_name]["one_hot"]
        json_data[field_name]["dtype"] = "int"
        json_data[field_name]["is_label"] = True
    if tag == "numeral":
        json_data[field_name]["one_hot"] = True
        json_data[field_name]["dtype"] = "float32"
        del json_data[field_name]["vocab_size"]
        json_data[field_name]["bucket_boundaries"] = []
    elif tag == "sequence":
        json_data[field_name]["embedding_combiner"] = "seq_pad"
        json_data[field_name]["val_sep"] = ","
        json_data[field_name]["max_len"] = 50
        json_data[field_name]["is_seq"] = True

json_data.update(old_data)
json_data = sort_json_by_order(json_data, field_names_valid)

for x in field_names_del:
    if x in json_data:
        del json_data[x]

with open(json_file_dd, 'w') as outfile:
    json.dump(json_data, outfile, indent=2)
