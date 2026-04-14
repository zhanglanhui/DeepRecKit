import json
import xml.etree.ElementTree as ET
# python
import json, sys
import xml.dom.minidom as minidom

if len(sys.argv) < 2:
    print("Please provide the path to the JSON file as a command line argument.")
    sys.exit(1)

json_file_path = sys.argv[1]

def json_to_xml(json_data):
    # data = json.loads(json_data)
    data = json_data

    doc = minidom.Document()

    root = doc.createElement("models")
    doc.appendChild(root)

    model = doc.createElement("model")
    root.appendChild(model)

    model_path = doc.createElement("model_path")
    model_path.appendChild(doc.createTextNode("/data/python_proj/banner/model/deepfm/model_saved"))
    model.appendChild(model_path)

    model_name = doc.createElement("model_name")
    model_name.appendChild(doc.createTextNode("deepfm_banner"))
    model.appendChild(model_name)

    prefix_name = doc.createElement("prefix_name")
    prefix_name.appendChild(doc.createTextNode("serving_default"))
    model.appendChild(prefix_name)

    prefix_name = doc.createElement("key_field")
    prefix_name.appendChild(doc.createTextNode("i_banner_id"))
    model.appendChild(prefix_name)

    inputs = doc.createElement("inputs")
    model.appendChild(inputs)
    user = doc.createElement("user")
    inputs.appendChild(user)
    for key, value in data.items():
        if str(key).startswith("i_") or value.get("is_label", False):
            continue
        field = doc.createElement("field")
        user.appendChild(field)
        name = doc.createElement("name")
        name.appendChild(doc.createTextNode(key))
        field.appendChild(name)
        for sub_key, sub_value in value.items():
            if sub_key in ("is_label", "is_seq", "hash_size", "emb_dim", "share_emb"):
                continue
            child = doc.createElement(sub_key)
            child.appendChild(doc.createTextNode(str(sub_value)))
            field.appendChild(child)

    item = doc.createElement("item")
    inputs.appendChild(item)
    for key, value in data.items():
        if not str(key).startswith("i_"):
            continue
        field = doc.createElement("field")
        item.appendChild(field)
        name = doc.createElement("name")
        name.appendChild(doc.createTextNode(key))
        field.appendChild(name)
        for sub_key, sub_value in value.items():
            if sub_key in ("is_label", "is_seq", "hash_size", "emb_dim", "share_emb"):
                continue
            child = doc.createElement(sub_key)
            child.appendChild(doc.createTextNode(str(sub_value)))
            field.appendChild(child)

    outputs = doc.createElement("outputs")
    model.appendChild(outputs)
    field = doc.createElement("field")
    outputs.appendChild(field)
    name = doc.createElement("name")
    name.appendChild(doc.createTextNode("StatefulPartitionedCall"))
    field.appendChild(name)
    dtype = doc.createElement("dtype")
    dtype.appendChild(doc.createTextNode("float"))
    field.appendChild(dtype)
    default1 = doc.createElement("default")
    default1.appendChild(doc.createTextNode("0"))
    field.appendChild(default1)

    xml_data = doc.toprettyxml(indent="    ")

    return xml_data

def write_xml_to_file(xml_data, file_path):
    with open(file_path, "w") as file:
        file.write(xml_data)
    print(f"XML data has been written to {file_path}")

with open(json_file_path, 'r', encoding='utf-8') as json_file:
    data = json.load(json_file)

xml_data = json_to_xml(data)
# print(xml_data)
write_xml_to_file(xml_data, "./model.xml")
