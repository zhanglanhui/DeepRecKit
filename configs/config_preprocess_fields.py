import json
import sys

if len(sys.argv) < 2:
    print("Please provide the path to the JSON file as a command line argument.")
    sys.exit(1)

json_file_path = sys.argv[1]

with open(json_file_path) as f:
    json_config = json.load(f)

# Build the schema used by DataInput when parsing raw records.
out = dict()
for name, x in json_config.items():
    tmp = dict()
    zscore = x.get('zscore', None)
    if name in ["label_ctr_long", "label_cvr_long"] or zscore:
        continue
    dtype111 = x['dtype']
    is_label = x.get('is_label', None)
    if dtype111 == 'float32':
        dtype = 'tf.float32'
        default_value = 0
    elif dtype111 in ['int32', 'int']:
        dtype = 'tf.int64'
        default_value = 0
    else:
        dtype = 'tf.string'
        default_value = '0'
    max_len = x.get('max_len', None)
    if max_len:
        tmp['shape'] = max_len
    else:
        tmp['shape'] = 1
    val_sep = x.get('val_sep', None)
    if val_sep:
        tmp['seperator'] = val_sep
        dtype = 'tf.int64'
        default_value = ''
        tmp['ftype'] = 'tf.VarLenFeature'
    else:
        tmp['ftype'] = 'tf.FixedLenFeature'
    if is_label:
        continue
    tmp['data_type'] = 'feature'
    tmp['dtype'] = dtype
    tmp['default_value'] = default_value
    tmp['essential'] = False
    out[name] = tmp

out["label"] = {
    'shape': 1,
    'ftype': 'tf.FixedLenFeature',
    'data_type': 'target',
    'dtype': 'tf.float32',
    'default_value': 0.0,
    'essential': True
}
print(json.dumps(out))
