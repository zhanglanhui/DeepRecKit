import json
import sys
import tensorflow as tf

if len(sys.argv) < 2:
    print("Please provide the path to the JSON file as a command line argument.")
    sys.exit(1)

json_file_path = sys.argv[1]

with open(json_file_path) as f:
    json_config = json.load(f)

out_list = []
for name, x in json_config.items():
    if "label" in name:
        continue
    dtype111 = x['dtype']
    max_len = x.get('max_len', None)
    bucket_boundaries = x.get('bucket_boundaries', None)
    if name and not max_len and bucket_boundaries and dtype111 in ["float", "int", "float32", "int64", "int32"]:
        out_list.append({
            'ftype': 'tf.feature_column.numeric_column',
            'input_feature_name': name,
            'output_feature_name': name + '_numeral',
            'parameters': {}
        })

# Bucketize dense numeric fields before they are embedded or crossed.
out_list1 = []
for name, x in json_config.items():
    if "label" in name:
        continue
    dtype111 = x['dtype']
    max_len = x.get('max_len', None)
    bucket_boundaries = x.get('bucket_boundaries', None)
    if name and not max_len and bucket_boundaries and dtype111 in ["float", "int", "float32", "int64", "int32"]:
        out_list1.append({
            'ftype': 'tf.feature_column.bucketized_column',
            'input_feature_name': name + '_numeral',
            'output_feature_name': name + '_numeral_bucket',
            'parameters': {
                'boundaries': bucket_boundaries
            }
        })

# Build sparse categorical columns from bounded-id or hashed-id fields.
out_list2 = []
for name, x in json_config.items():
    if "label" in name:
        continue
    dtype111 = x['dtype']
    vocab_size = x.get('vocab_size', None)
    if name and vocab_size:
        if "u_uin" == name:
            out_list2.append({
                'ftype': 'tf.feature_column.categorical_column_with_hash_bucket',
                'input_feature_name': name,
                'output_feature_name': name + '_category',
                'parameters': {
                    'hash_bucket_size': vocab_size
                }
            })
        else:
            out_list2.append({
                'ftype': 'tf.feature_column.categorical_column_with_identity',
                'input_feature_name': name,
                'output_feature_name': name + '_category',
                'parameters': {
                    'num_buckets': vocab_size
                }
            })

# Turn categorical columns into embedding columns used by deep models.
out_list3 = []
out_map3 = dict()
for name, x in json_config.items():
    if "label" in name:
        continue
    vocab_size = x.get('vocab_size', None)
    embedding_dim = x.get('emb_dim', None)
    embedding_name = x.get('embedding_name', None)
    if name and vocab_size and embedding_dim:
        if not embedding_name:
            out_list3.append({
                'ftype': 'tf.feature_column.embedding_column',
                'input_feature_name': name + '_category',
                'output_feature_name': name + '_category_embedding',
                'parameters': {
                    'dimension': embedding_dim,
                    'combiner': 'sum'
                }
            })
        else:
            if embedding_name in out_map3:
                out_map3[embedding_name]["input_feature_name"].append(name + "_category")
                out_map3[embedding_name]["output_feature_name"].append(name + "_category_shared_embedding")
            else:
                out_map3[embedding_name] = {
                    'ftype': 'tf.feature_column.shared_embedding_columns',
                    'input_feature_name': [name + '_category'],
                    'output_feature_name': [name + '_category_shared_embedding'],
                    'parameters': {
                        'dimension': embedding_dim,
                        'combiner': 'sum'
                    }
                }
        if name not in ["u_uin"] and vocab_size <= 0:
            out_list3.append({
                'ftype': 'tf.feature_column.indicator_column',
                'input_feature_name': name + '_category',
                'output_feature_name': name + '_category_indicator',
                'parameters': {}
            })

out_list33 = []
for _, v in out_map3.items():
    out_list33.append(v)

# Bucketized numeric fields can also participate in the deep path via embeddings.
out_list4 = []
for name, x in json_config.items():
    if "label" in name:
        continue
    dtype111 = x['dtype']
    max_len = x.get('max_len', None)
    bucket_boundaries = x.get('bucket_boundaries', None)
    if name and not max_len and bucket_boundaries and dtype111 in ["float", "int", "float32", "int64", "int32"]:
        out_list4.append({
            'ftype': 'tf.feature_column.embedding_column',
            'input_feature_name': name + '_numeral_bucket',
            'output_feature_name': name + '_numeral_bucket_embedding',
            'parameters': {
                'dimension': 16,
                'combiner': 'sum'
            }
        })

# Add crossed columns for related sparse fields.
out_list5 = []
crossed_list = set()
for name, x in json_config.items():
    if "label" in name:
        continue
    dtype111 = x['dtype']
    vocab_size = x.get('vocab_size', None)
    embedding_name = x.get('embedding_name', None)
    if name and vocab_size and embedding_name and dtype111 in ["float", "int", "float32", "int64", "int32"]:
        cross_name = embedding_name[:-4]
        if name == cross_name:
            continue
        out_list5.append({
            'ftype': 'tf.feature_column.crossed_column',
            'input_feature_name': [name, cross_name],
            'output_feature_name': name + "$" + cross_name + '_category',
            'parameters': {
                'hash_bucket_size': 10 * vocab_size
            }
        })
        crossed_list.add(name + "$" + cross_name)

sp_cross = []
for fields in sp_cross:
    if "$".join(fields) in crossed_list or "$".join(fields[::-1]) in crossed_list:
        continue
    size1 = json_config[fields[0]]['vocab_size']
    size2 = json_config[fields[1]]['vocab_size']
    vocab_size = max(size1, size2)
    out_list5.append({
        'ftype': 'tf.feature_column.crossed_column',
        'input_feature_name': fields,
        'output_feature_name': "$".join(fields) + '_category',
        'parameters': {
            'hash_bucket_size': vocab_size
        }
    })

out = dict()
field_struct = []
field_struct.extend(out_list)
field_struct.extend(out_list1)
field_struct.extend(out_list2)
field_struct.extend(out_list3)
field_struct.extend(out_list33)
field_struct.extend(out_list4)
field_struct.extend(out_list5)

out["feature_column_config_list"] = field_struct
column_group = dict()
output_feature_name = [x["output_feature_name"] for x in field_struct if isinstance(x["output_feature_name"], str)]
for x in field_struct:
    if isinstance(x["output_feature_name"], list):
        output_feature_name.extend(x["output_feature_name"])

column_group["wide"] = [x for x in output_feature_name if "_can_" not in x and x.endswith("_category")]
column_group["deep"] = [x for x in output_feature_name if "_can_" not in x and x.endswith("_embedding")]
column_group["cross"] = [x for x in output_feature_name if x.endswith("_embedding") and "_can_" not in x]
column_group["fm"] = [x for x in output_feature_name if x.endswith("_embedding") and "_can_" not in x]
column_group["attention_1"] = []
column_group["attention_2"] = []
column_group["attention_3"] = []
column_group["attention_4"] = []
column_group["pnn_1"] = []
column_group["pnn_2"] = []
column_group["pnn_3"] = []
column_group["pnn_4"] = []

out["feature_column_group"] = column_group

print(json.dumps(out))
