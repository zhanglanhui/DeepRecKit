import tensorflow as tf
import numpy as np

dataset = tf.data.Dataset.from_tensor_slices(
    {
        "a": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        "b": np.random.uniform(size=(5, 2))
    }
)
# tf.data.TextLineDataset()
# tf.data.Dataset.

iterator = dataset.make_one_shot_iterator()
one_element = iterator.get_next()
with tf.Session() as sess:
    for i in range(5):
        print(sess.run(one_element))



import pandas as pd
df = pd.read_csv('part-00000-c572590b-04ad-4039-9d66-19512ab13464-c000.csv' ,_default_sep="'$'")

import os
import pandas as pd

def get_dir_file_name(file_dir, suffix=".txt"):
    L = []
    if suffix:
        for files in os.listdir(file_dir):
            file_path = os.path.join(file_dir, files)
            if os.path.isfile(file_path) and os.path.splitext(file_path)[1] == suffix:
                L.append(file_path)
    else:
        for files in os.listdir(file_dir):
            file_path = os.path.join(file_dir, files)
            if os.path.isfile(file_path):
                L.append(file_path)
    return L

def get_multi_csv_data(file_path, max_len=-1):
    file_num = 0
    frames = []
    for x in get_dir_file_name(file_path, suffix=".csv"):
        frames.append(pd.read_csv(x,sep="$"))
        file_num += 1
        if file_num >= max_len:
            break
    result = pd.concat(frames)
    return result
files.get_multi_csv_data("./csv",max_len=10)
