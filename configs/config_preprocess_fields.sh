#/bin/bash
base_path=$(pwd)/configs/
config_file=$base_path"config_preprocess_fields.json"
conf_path=$1
python $base_path"config_preprocess_fields.py" $conf_path | python -m json.tool > $config_file

sed -i 's/false/False/g' $config_file
sed -i 's/true/True/g' $config_file

