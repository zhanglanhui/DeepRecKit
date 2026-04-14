import tensorflow as tf
from sklearn.model_selection import train_test_split
import pandas as pd
from pyutils.files import FilesOp
import numpy as np

def float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def serialize(writer, dataes: pd.DataFrame, int64_list, float_list, bytes_list, vector_list):
    for index, data in dataes.iterrows():
        int64_features = {x: int64_feature(int(data.get(x))) for x in int64_list}
        float_features = {x: float_feature(float(data.get(x))) for x in float_list}
        bytes_features = {x: bytes_feature(str(data.get(x)).encode('utf-8')) for x in bytes_list}
        vector_features= {x: bytes_feature(np.array(map(int,str(data.get(x)).split(","))).tostring()) for x in vector_list}
        dictMerged = dict()
        dictMerged.update(int64_features)
        dictMerged.update(float_features)
        dictMerged.update(bytes_features)
        dictMerged.update(vector_features)
        # print("\ndictMerged\n", dictMerged["label"])
        # print("\ndata\n", type(channel))
        """ 2. 瀹氫箟features """
        example = tf.train.Example(
            features=tf.train.Features(
                feature=dictMerged))

        """ 3. 搴忓垪鍖?鍐欏叆"""
        serialized = example.SerializeToString()
        writer.write(serialized)


# int64_list = ["label", "item_id", "content_size"]
int64_list=["label", "item_id", "content_size"]
# float_list = ["collect_cnt", "view_cnt", "item_click_sparse_n7d_stat_behavior_cnt_all",
#               "item_click_sparse_n7d_stat_user_cnt_all", "item_click_sparse_n7d_stat_user_avg_all",
#               "item_read_sparse_n7d_stat_behavior_cnt_all",
#               "item_read_sparse_n7d_stat_user_cnt_all", "item_read_sparse_n7d_stat_user_avg_all",
#               "item_click_sparse_n7d_stat_behavior_ctr_all", "item_click_sparse_n7d_stat_user_ctr_all"]
float_list = []
bytes_list = ["platform", "device_model", "device_brand", "net_type", "login", "user_id_type", "sex", "gender",
              "channel", "user_channel","user_key"]
vector_list = ["hist_items"]

options_gzip = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
writer_train = tf.io.TFRecordWriter("./data/samples/train.bin1", options=options_gzip)
writer_eval = tf.io.TFRecordWriter("./data/eval/eval.bin1", options=options_gzip)
# dataes = pd.read_csv("../deepFM_test/xxx.csv", nrows=100)
dataes = FilesOp().get_multi_csv_data("/data/airec/data/csv/", max_len=70, sep="$")
# dataes = pd.read_csv("../deepfm/part-00000-85ccdfbd-42af-4fd8-bcd6-819d2428db63-c000.csv")
dataes = dataes[int64_list + float_list + bytes_list]
dataes = dataes.dropna()
print("dataes length:", dataes.count())
print("dataes dtypes:", dataes.dtypes)
df_train, df_test = train_test_split(dataes, test_size=0.2, stratify=dataes['label'])
serialize(writer_train, df_train, int64_list, float_list, bytes_list,vector_list )
serialize(writer_eval, df_test, int64_list, float_list, bytes_list, vector_list)

# tags

