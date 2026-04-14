#/bin/bash
base_path=$(pwd)/configs/
config_file=$base_path"config_model_struct.json"
conf_path=$1
python $base_path"config_model_struct.py" $conf_path | python -m json.tool > $config_file
#python $base_path"config_model_struct.py" $conf_path

sed -i 's/\"tf.feature_column.numeric_column\"/tf.feature_column.numeric_column/g' $config_file
sed -i 's/\"tf.feature_column.indicator_column\"/tf.feature_column.indicator_column/g' $config_file
sed -i 's/\"tf.feature_column.bucketized_column\"/tf.feature_column.bucketized_column/g' $config_file
sed -i 's/\"tf.feature_column.categorical_column_with_hash_bucket\"/tf.feature_column.categorical_column_with_hash_bucket/g' $config_file
sed -i 's/\"tf.feature_column.categorical_column_with_identity\"/tf.feature_column.categorical_column_with_identity/g' $config_file
sed -i 's/\"tf.feature_column.embedding_column\"/tf.feature_column.embedding_column/g' $config_file
sed -i 's/\"tf.feature_column.crossed_column\"/tf.feature_column.crossed_column/g' $config_file
sed -i 's/\"tf.feature_column.shared_embedding_columns\"/tf.feature_column.shared_embedding_columns/g' $config_file

