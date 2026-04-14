#!/bin/sh

conf_path=$(pwd)/data/open_data/criteo_sample/criteo_sample_feature_config_bucketized.json

# Generate the feature transform graph and the input parsing schema.
sh configs/config_model_struct.sh $conf_path
sh configs/config_preprocess_fields.sh $conf_path
sh configs/config_train_fields.sh $conf_path

# Write the training-side feature config and transform config into Python modules.
echo "import tensorflow as tf" > feature_config/model_input_fn.py
echo "" >> feature_config/model_input_fn.py
echo -n "FEATURE_CONFIG = " >> feature_config/model_input_fn.py
cat configs/config_train_fields.json >> feature_config/model_input_fn.py
echo -n "FEATURE_TRANSFORM_CONFIG = " >> feature_config/model_input_fn.py
cat configs/config_model_struct.json >> feature_config/model_input_fn.py

# Write the generated schema used by DataInput when parsing raw CSV records.
echo -n "FEATURE_CONFIG = " > feature_config/model_json_conf.py
cat configs/config_preprocess_fields.json >> feature_config/model_json_conf.py
