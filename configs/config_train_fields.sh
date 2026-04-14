#/bin/bash
base_path=$(pwd)/configs/
config_file=$base_path"config_train_fields.json"
conf_path=$1
python $base_path"config_train_fields.py" $conf_path   | python -m json.tool > $config_file

sed -i 's/false/False/g' $config_file
sed -i 's/true/True/g' $config_file
sed -i 's/\"tf.FixedLenFeature\"/tf.FixedLenFeature/g' $config_file
sed -i 's/\"tf.VarLenFeature\"/tf.VarLenFeature/g' $config_file
sed -i 's/\"tf.int64\"/tf.int64/g' $config_file
sed -i 's/\"tf.string\"/tf.string/g' $config_file
sed -i 's/\"tf.float32\"/tf.float32/g' $config_file
