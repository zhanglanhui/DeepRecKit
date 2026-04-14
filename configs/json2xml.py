# !/usr/bin/python
# -*- coding: UTF-8 -*-
"""
Created on MAY 21, 2018
@author: zlh
"""

import json
import xml.etree.ElementTree as ET

    i = "\n" + level * "\t"
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "\t"
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            __indent(elem, level + 1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i

def trans_data_type(dtype):
    # if dtype == "int32":
    #     return "int"
    # elif dtype == "int64":
    #     return "long"
    # elif dtype == "float32":
    #     return "float"
    # elif dtype == "float64":
    #     return "double"
    if dtype == "float32" or dtype == "float64":
        return "float"
    else:
        return "string"

if __name__ == "__main__":
    """
        Usage: pi [partitions]
    """
    import argparse

    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument("--alg", type=str, default='serve')
    parser.add_argument("--json_path", type=str,
                        default='./model_usered.json')
    parser.add_argument("--xml_path", type=str,
                        default='model_desc.xml')
    args = parser.parse_args()
    print(args)

    with open(args.json_path, "r") as f:
        js = json.load(f)
        # print("ss", js["inputs"])

    root = ET.Element('configuration')
    # Add sub element.
    models = ET.SubElement(root, "models")
    tree = ET.ElementTree(root)

    # Add sub element.
    model = ET.SubElement(models, "model")
    tree1 = ET.ElementTree(root)

    # model tags
    # model_tags = ET.SubElement(model, "model_tags")
    model_tag = ET.SubElement(model, "model_name")
    model_tag.text = "wdcc_banner"
    model_path = ET.SubElement(model, "model_path")
    model_path.text = "/data/nfs/wdcc_banner"

    model_hdfs_path = ET.SubElement(model, "key_field")
    model_hdfs_path.text = "i_banner_id"

    item_fea = ET.SubElement(model, "item_features")
    user_fea = ET.SubElement(model, "user_features")

    for name, item in js.items():
        # print(item)
        if item.get("is_label", None) is True:
            continue
        # name = item.get("name", None)
        dtype = item.get("dtype", None)
        zscore = item.get("zscore", None)
        mean = item.get("mean", None)
        val_sep = item.get("val_sep", None)
        if name and dtype:
            if name.startswith("u") or name.startswith("realtime_") or name.startswith("c_"):
                fea = ET.SubElement(user_fea, "feature")
                if val_sep:
                    fea.text = name + "," + trans_data_type(dtype) + ","
                else:
                    fea.text = name + "," + trans_data_type(dtype) + ",0"
            elif name.startswith("i"):
                fea = ET.SubElement(item_fea, "feature")
                if val_sep:
                    fea.text = name + "," + trans_data_type(dtype) + ","
                else:
                    fea.text = name + "," + trans_data_type(dtype) + ",0"
            else:
                print("name error:", name)
                pass
    # feature_prefix
    serving = ET.SubElement(model, "model_tags")
    output_name = ET.SubElement(serving, "model_tag")
    output_name.text = 'serve'

    # output = ET.SubElement(outputs, "output")
    output_name = ET.SubElement(outputs, "output")
    output_name.text = 'Sigmoid_1,float,1'
    output_name = ET.SubElement(outputs, "output")
    output_name.text = 'Sigmoid_2,float,1'
    output_name = ET.SubElement(outputs, "output")
    output_name.text = 'Sigmoid_3,float,1'

    __indent(root)
