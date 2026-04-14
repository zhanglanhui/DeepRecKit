#!/usr/bin/python
# -*- coding: UTF-8 -*-
import pandas as pd
import tensorflow as tf

def read_text_file(text_file):
    lines = []
    with open(text_file, "r") as f:
        for line in f:
            lines.append(line.strip())
    return lines

def float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def serialize(writer, dataes: pd.DataFrame, int64_list, float_list, bytes_list, ):
    for index, data in dataes.iterrows():
        int64_features = {x: int64_feature(int(data.get(x))) for x in int64_list}
        float_features = {x: float_feature(float(data.get(x))) for x in float_list}
        bytes_features = {x: bytes_feature(str(data.get(x)).encode('utf-8')) for x in bytes_list}
        dictMerged = dict()
        dictMerged.update(int64_features)
        dictMerged.update(float_features)
        dictMerged.update(bytes_features)
        # print("\ndata\n", data)
        # print("\ndata\n", type(channel))
        """ 2. 瀹氫箟features """
        example = tf.train.Example(
            features=tf.train.Features(
                feature=dictMerged))

        """ 3. 搴忓垪鍖?鍐欏叆"""
        serialized = example.SerializeToString()
        writer.write(serialized)

# platform                                          object
# device_model                                      object
# device_brand                                      object
# day                                               object
# label                                              int64
# weight                                             int64
# weekday                                            int64
# vacation                                           int64
# author                                            object
# channel                                           object
# collect_cnt                                        int64
# color_id                                           int64
# copyright_type                                     int64
# gender                                             int64
# tags                                              object
# view_cnt                                           int64
# vip                                                int64
# item_click_sparse_n7d_stat_behavior_cnt_all        int64
# item_click_sparse_n7d_stat_user_cnt_all            int64
# item_click_sparse_n7d_stat_user_avg_all          float64
# item_collect_sparse_n7d_stat_behavior_cnt_all      int64
# item_collect_sparse_n7d_stat_user_cnt_all          int64
# item_collect_sparse_n7d_stat_user_avg_all        float64
# item_read_sparse_n7d_stat_behavior_cnt_all         int64
# item_read_sparse_n7d_stat_user_cnt_all             int64
# item_read_sparse_n7d_stat_user_avg_all           float64
# item_click_sparse_n7d_stat_behavior_ctr_all      float64
# item_click_sparse_n7d_stat_user_ctr_all          float64
# sex                                                int64
# user_channel                                      object
# user_read_dense_l7d_dist_channel_score_all        object
# user_collect_dense_l7d_dist_channel_score_all    float64
# user_click_dense_l7d_dist_channel_score_all       object
# user_comment_dense_l7d_dist_channel_score_all     object


# int64_list = ["label"]
# float_list = ["collect_cnt", "view_cnt", "item_click_sparse_n7d_stat_behavior_cnt_all",
#               "item_click_sparse_n7d_stat_user_cnt_all", "item_click_sparse_n7d_stat_user_avg_all",
#               "item_collect_sparse_n7d_stat_behavior_cnt_all", "item_collect_sparse_n7d_stat_user_cnt_all",
#               "item_collect_sparse_n7d_stat_user_avg_all", "item_read_sparse_n7d_stat_behavior_cnt_all",
#               "item_read_sparse_n7d_stat_user_cnt_all", "item_read_sparse_n7d_stat_user_avg_all",
#               "item_click_sparse_n7d_stat_behavior_ctr_all", "item_click_sparse_n7d_stat_user_ctr_all"]
# bytes_list = ["user_key", "item_id", "item_type", "platform", "device_model", "device_brand", "weekday", "vacation",
#               "author", "channel", "color_id", "copyright_type", "gender", "vip", "sex",
#               "user_channel"]
# # dataes = pd.read_csv("../deepFM_test/xxx.csv", nrows=100)
# dataes = pd.read_csv("../train.csv")
# dataes = dataes[int64_list + float_list + bytes_list]
# dataes = dataes.dropna()
# print("dataes length:", dataes.count())
# print("dataes dtypes:", dataes.dtypes)
# df_train, df_test = train_test_split(dataes, test_size=0.2, stratify=dataes['label'])
# serialize(writer_train, df_train, int64_list, float_list, bytes_list, )
# serialize(writer_eval, df_test, int64_list, float_list, bytes_list, )
