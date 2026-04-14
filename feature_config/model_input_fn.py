import tensorflow as tf

FEATURE_CONFIG = {
    "I1": {
        "shape": 1,
        "ftype": tf.FixedLenFeature,
        "data_type": "feature",
        "dtype": tf.float32,
        "default_value": 0,
        "essential": False
    },
    "I2": {
        "shape": 1,
        "ftype": tf.FixedLenFeature,
        "data_type": "feature",
        "dtype": tf.float32,
        "default_value": 0,
        "essential": False
    },
    "I3": {
        "shape": 1,
        "ftype": tf.FixedLenFeature,
        "data_type": "feature",
        "dtype": tf.float32,
        "default_value": 0,
        "essential": False
    },
    "I4": {
        "shape": 1,
        "ftype": tf.FixedLenFeature,
        "data_type": "feature",
        "dtype": tf.float32,
        "default_value": 0,
        "essential": False
    },
    "I5": {
        "shape": 1,
        "ftype": tf.FixedLenFeature,
        "data_type": "feature",
        "dtype": tf.float32,
        "default_value": 0,
        "essential": False
    },
    "I6": {
        "shape": 1,
        "ftype": tf.FixedLenFeature,
        "data_type": "feature",
        "dtype": tf.float32,
        "default_value": 0,
        "essential": False
    },
    "I7": {
        "shape": 1,
        "ftype": tf.FixedLenFeature,
        "data_type": "feature",
        "dtype": tf.float32,
        "default_value": 0,
        "essential": False
    },
    "I8": {
        "shape": 1,
        "ftype": tf.FixedLenFeature,
        "data_type": "feature",
        "dtype": tf.float32,
        "default_value": 0,
        "essential": False
    },
    "I9": {
        "shape": 1,
        "ftype": tf.FixedLenFeature,
        "data_type": "feature",
        "dtype": tf.float32,
        "default_value": 0,
        "essential": False
    },
    "I10": {
        "shape": 1,
        "ftype": tf.FixedLenFeature,
        "data_type": "feature",
        "dtype": tf.float32,
        "default_value": 0,
        "essential": False
    },
    "I11": {
        "shape": 1,
        "ftype": tf.FixedLenFeature,
        "data_type": "feature",
        "dtype": tf.float32,
        "default_value": 0,
        "essential": False
    },
    "I12": {
        "shape": 1,
        "ftype": tf.FixedLenFeature,
        "data_type": "feature",
        "dtype": tf.float32,
        "default_value": 0,
        "essential": False
    },
    "I13": {
        "shape": 1,
        "ftype": tf.FixedLenFeature,
        "data_type": "feature",
        "dtype": tf.float32,
        "default_value": 0,
        "essential": False
    },
    "C1": {
        "shape": 1,
        "ftype": tf.FixedLenFeature,
        "data_type": "feature",
        "dtype": tf.int64,
        "default_value": "0",
        "essential": False
    },
    "C2": {
        "shape": 1,
        "ftype": tf.FixedLenFeature,
        "data_type": "feature",
        "dtype": tf.int64,
        "default_value": "0",
        "essential": False
    },
    "C3": {
        "shape": 1,
        "ftype": tf.FixedLenFeature,
        "data_type": "feature",
        "dtype": tf.int64,
        "default_value": "0",
        "essential": False
    },
    "C4": {
        "shape": 1,
        "ftype": tf.FixedLenFeature,
        "data_type": "feature",
        "dtype": tf.int64,
        "default_value": "0",
        "essential": False
    },
    "C5": {
        "shape": 1,
        "ftype": tf.FixedLenFeature,
        "data_type": "feature",
        "dtype": tf.int64,
        "default_value": "0",
        "essential": False
    },
    "C6": {
        "shape": 1,
        "ftype": tf.FixedLenFeature,
        "data_type": "feature",
        "dtype": tf.int64,
        "default_value": "0",
        "essential": False
    },
    "C7": {
        "shape": 1,
        "ftype": tf.FixedLenFeature,
        "data_type": "feature",
        "dtype": tf.int64,
        "default_value": "0",
        "essential": False
    },
    "C8": {
        "shape": 1,
        "ftype": tf.FixedLenFeature,
        "data_type": "feature",
        "dtype": tf.int64,
        "default_value": "0",
        "essential": False
    },
    "C9": {
        "shape": 1,
        "ftype": tf.FixedLenFeature,
        "data_type": "feature",
        "dtype": tf.int64,
        "default_value": "0",
        "essential": False
    },
    "C10": {
        "shape": 1,
        "ftype": tf.FixedLenFeature,
        "data_type": "feature",
        "dtype": tf.int64,
        "default_value": "0",
        "essential": False
    },
    "C11": {
        "shape": 1,
        "ftype": tf.FixedLenFeature,
        "data_type": "feature",
        "dtype": tf.int64,
        "default_value": "0",
        "essential": False
    },
    "C12": {
        "shape": 1,
        "ftype": tf.FixedLenFeature,
        "data_type": "feature",
        "dtype": tf.int64,
        "default_value": "0",
        "essential": False
    },
    "C13": {
        "shape": 1,
        "ftype": tf.FixedLenFeature,
        "data_type": "feature",
        "dtype": tf.int64,
        "default_value": "0",
        "essential": False
    },
    "C14": {
        "shape": 1,
        "ftype": tf.FixedLenFeature,
        "data_type": "feature",
        "dtype": tf.int64,
        "default_value": "0",
        "essential": False
    },
    "C15": {
        "shape": 1,
        "ftype": tf.FixedLenFeature,
        "data_type": "feature",
        "dtype": tf.int64,
        "default_value": "0",
        "essential": False
    },
    "C16": {
        "shape": 1,
        "ftype": tf.FixedLenFeature,
        "data_type": "feature",
        "dtype": tf.int64,
        "default_value": "0",
        "essential": False
    },
    "C17": {
        "shape": 1,
        "ftype": tf.FixedLenFeature,
        "data_type": "feature",
        "dtype": tf.int64,
        "default_value": "0",
        "essential": False
    },
    "C18": {
        "shape": 1,
        "ftype": tf.FixedLenFeature,
        "data_type": "feature",
        "dtype": tf.int64,
        "default_value": "0",
        "essential": False
    },
    "C19": {
        "shape": 1,
        "ftype": tf.FixedLenFeature,
        "data_type": "feature",
        "dtype": tf.int64,
        "default_value": "0",
        "essential": False
    },
    "C20": {
        "shape": 1,
        "ftype": tf.FixedLenFeature,
        "data_type": "feature",
        "dtype": tf.int64,
        "default_value": "0",
        "essential": False
    },
    "C21": {
        "shape": 1,
        "ftype": tf.FixedLenFeature,
        "data_type": "feature",
        "dtype": tf.int64,
        "default_value": "0",
        "essential": False
    },
    "C22": {
        "shape": 1,
        "ftype": tf.FixedLenFeature,
        "data_type": "feature",
        "dtype": tf.int64,
        "default_value": "0",
        "essential": False
    },
    "C23": {
        "shape": 1,
        "ftype": tf.FixedLenFeature,
        "data_type": "feature",
        "dtype": tf.int64,
        "default_value": "0",
        "essential": False
    },
    "C24": {
        "shape": 1,
        "ftype": tf.FixedLenFeature,
        "data_type": "feature",
        "dtype": tf.int64,
        "default_value": "0",
        "essential": False
    },
    "C25": {
        "shape": 1,
        "ftype": tf.FixedLenFeature,
        "data_type": "feature",
        "dtype": tf.int64,
        "default_value": "0",
        "essential": False
    },
    "C26": {
        "shape": 1,
        "ftype": tf.FixedLenFeature,
        "data_type": "feature",
        "dtype": tf.int64,
        "default_value": "0",
        "essential": False
    }
}
FEATURE_TRANSFORM_CONFIG = {
    "feature_column_config_list": [
        {
            "ftype": tf.feature_column.numeric_column,
            "input_feature_name": "I1",
            "output_feature_name": "I1_numeral",
            "parameters": {}
        },
        {
            "ftype": tf.feature_column.numeric_column,
            "input_feature_name": "I2",
            "output_feature_name": "I2_numeral",
            "parameters": {}
        },
        {
            "ftype": tf.feature_column.numeric_column,
            "input_feature_name": "I3",
            "output_feature_name": "I3_numeral",
            "parameters": {}
        },
        {
            "ftype": tf.feature_column.numeric_column,
            "input_feature_name": "I4",
            "output_feature_name": "I4_numeral",
            "parameters": {}
        },
        {
            "ftype": tf.feature_column.numeric_column,
            "input_feature_name": "I5",
            "output_feature_name": "I5_numeral",
            "parameters": {}
        },
        {
            "ftype": tf.feature_column.numeric_column,
            "input_feature_name": "I6",
            "output_feature_name": "I6_numeral",
            "parameters": {}
        },
        {
            "ftype": tf.feature_column.numeric_column,
            "input_feature_name": "I7",
            "output_feature_name": "I7_numeral",
            "parameters": {}
        },
        {
            "ftype": tf.feature_column.numeric_column,
            "input_feature_name": "I8",
            "output_feature_name": "I8_numeral",
            "parameters": {}
        },
        {
            "ftype": tf.feature_column.numeric_column,
            "input_feature_name": "I9",
            "output_feature_name": "I9_numeral",
            "parameters": {}
        },
        {
            "ftype": tf.feature_column.numeric_column,
            "input_feature_name": "I10",
            "output_feature_name": "I10_numeral",
            "parameters": {}
        },
        {
            "ftype": tf.feature_column.numeric_column,
            "input_feature_name": "I11",
            "output_feature_name": "I11_numeral",
            "parameters": {}
        },
        {
            "ftype": tf.feature_column.numeric_column,
            "input_feature_name": "I12",
            "output_feature_name": "I12_numeral",
            "parameters": {}
        },
        {
            "ftype": tf.feature_column.numeric_column,
            "input_feature_name": "I13",
            "output_feature_name": "I13_numeral",
            "parameters": {}
        },
        {
            "ftype": tf.feature_column.bucketized_column,
            "input_feature_name": "I1_numeral",
            "output_feature_name": "I1_numeral_bucket",
            "parameters": {
                "boundaries": [
                    0,
                    0.001,
                    0.005,
                    0.01,
                    0.02,
                    0.05,
                    0.1,
                    0.2,
                    0.3,
                    0.4,
                    0.5,
                    0.6,
                    0.7,
                    0.8,
                    0.9,
                    1.0
                ]
            }
        },
        {
            "ftype": tf.feature_column.bucketized_column,
            "input_feature_name": "I2_numeral",
            "output_feature_name": "I2_numeral_bucket",
            "parameters": {
                "boundaries": [
                    0,
                    0.001,
                    0.005,
                    0.01,
                    0.02,
                    0.05,
                    0.1,
                    0.2,
                    0.3,
                    0.4,
                    0.5,
                    0.6,
                    0.7,
                    0.8,
                    0.9,
                    1.0
                ]
            }
        },
        {
            "ftype": tf.feature_column.bucketized_column,
            "input_feature_name": "I3_numeral",
            "output_feature_name": "I3_numeral_bucket",
            "parameters": {
                "boundaries": [
                    0,
                    0.001,
                    0.005,
                    0.01,
                    0.02,
                    0.05,
                    0.1,
                    0.2,
                    0.3,
                    0.4,
                    0.5,
                    0.6,
                    0.7,
                    0.8,
                    0.9,
                    1.0
                ]
            }
        },
        {
            "ftype": tf.feature_column.bucketized_column,
            "input_feature_name": "I4_numeral",
            "output_feature_name": "I4_numeral_bucket",
            "parameters": {
                "boundaries": [
                    0,
                    0.001,
                    0.005,
                    0.01,
                    0.02,
                    0.05,
                    0.1,
                    0.2,
                    0.3,
                    0.4,
                    0.5,
                    0.6,
                    0.7,
                    0.8,
                    0.9,
                    1.0
                ]
            }
        },
        {
            "ftype": tf.feature_column.bucketized_column,
            "input_feature_name": "I5_numeral",
            "output_feature_name": "I5_numeral_bucket",
            "parameters": {
                "boundaries": [
                    0,
                    0.001,
                    0.005,
                    0.01,
                    0.02,
                    0.05,
                    0.1,
                    0.2,
                    0.3,
                    0.4,
                    0.5,
                    0.6,
                    0.7,
                    0.8,
                    0.9,
                    1.0
                ]
            }
        },
        {
            "ftype": tf.feature_column.bucketized_column,
            "input_feature_name": "I6_numeral",
            "output_feature_name": "I6_numeral_bucket",
            "parameters": {
                "boundaries": [
                    0,
                    0.001,
                    0.005,
                    0.01,
                    0.02,
                    0.05,
                    0.1,
                    0.2,
                    0.3,
                    0.4,
                    0.5,
                    0.6,
                    0.7,
                    0.8,
                    0.9,
                    1.0
                ]
            }
        },
        {
            "ftype": tf.feature_column.bucketized_column,
            "input_feature_name": "I7_numeral",
            "output_feature_name": "I7_numeral_bucket",
            "parameters": {
                "boundaries": [
                    0,
                    0.001,
                    0.005,
                    0.01,
                    0.02,
                    0.05,
                    0.1,
                    0.2,
                    0.3,
                    0.4,
                    0.5,
                    0.6,
                    0.7,
                    0.8,
                    0.9,
                    1.0
                ]
            }
        },
        {
            "ftype": tf.feature_column.bucketized_column,
            "input_feature_name": "I8_numeral",
            "output_feature_name": "I8_numeral_bucket",
            "parameters": {
                "boundaries": [
                    0,
                    0.001,
                    0.005,
                    0.01,
                    0.02,
                    0.05,
                    0.1,
                    0.2,
                    0.3,
                    0.4,
                    0.5,
                    0.6,
                    0.7,
                    0.8,
                    0.9,
                    1.0
                ]
            }
        },
        {
            "ftype": tf.feature_column.bucketized_column,
            "input_feature_name": "I9_numeral",
            "output_feature_name": "I9_numeral_bucket",
            "parameters": {
                "boundaries": [
                    0,
                    0.001,
                    0.005,
                    0.01,
                    0.02,
                    0.05,
                    0.1,
                    0.2,
                    0.3,
                    0.4,
                    0.5,
                    0.6,
                    0.7,
                    0.8,
                    0.9,
                    1.0
                ]
            }
        },
        {
            "ftype": tf.feature_column.bucketized_column,
            "input_feature_name": "I10_numeral",
            "output_feature_name": "I10_numeral_bucket",
            "parameters": {
                "boundaries": [
                    0,
                    0.001,
                    0.005,
                    0.01,
                    0.02,
                    0.05,
                    0.1,
                    0.2,
                    0.3,
                    0.4,
                    0.5,
                    0.6,
                    0.7,
                    0.8,
                    0.9,
                    1.0
                ]
            }
        },
        {
            "ftype": tf.feature_column.bucketized_column,
            "input_feature_name": "I11_numeral",
            "output_feature_name": "I11_numeral_bucket",
            "parameters": {
                "boundaries": [
                    0,
                    0.001,
                    0.005,
                    0.01,
                    0.02,
                    0.05,
                    0.1,
                    0.2,
                    0.3,
                    0.4,
                    0.5,
                    0.6,
                    0.7,
                    0.8,
                    0.9,
                    1.0
                ]
            }
        },
        {
            "ftype": tf.feature_column.bucketized_column,
            "input_feature_name": "I12_numeral",
            "output_feature_name": "I12_numeral_bucket",
            "parameters": {
                "boundaries": [
                    0,
                    0.001,
                    0.005,
                    0.01,
                    0.02,
                    0.05,
                    0.1,
                    0.2,
                    0.3,
                    0.4,
                    0.5,
                    0.6,
                    0.7,
                    0.8,
                    0.9,
                    1.0
                ]
            }
        },
        {
            "ftype": tf.feature_column.bucketized_column,
            "input_feature_name": "I13_numeral",
            "output_feature_name": "I13_numeral_bucket",
            "parameters": {
                "boundaries": [
                    0,
                    0.001,
                    0.005,
                    0.01,
                    0.02,
                    0.05,
                    0.1,
                    0.2,
                    0.3,
                    0.4,
                    0.5,
                    0.6,
                    0.7,
                    0.8,
                    0.9,
                    1.0
                ]
            }
        },
        {
            "ftype": tf.feature_column.categorical_column_with_identity,
            "input_feature_name": "C1",
            "output_feature_name": "C1_category",
            "parameters": {
                "num_buckets": 3000000
            }
        },
        {
            "ftype": tf.feature_column.categorical_column_with_identity,
            "input_feature_name": "C2",
            "output_feature_name": "C2_category",
            "parameters": {
                "num_buckets": 3000000
            }
        },
        {
            "ftype": tf.feature_column.categorical_column_with_identity,
            "input_feature_name": "C3",
            "output_feature_name": "C3_category",
            "parameters": {
                "num_buckets": 3000000
            }
        },
        {
            "ftype": tf.feature_column.categorical_column_with_identity,
            "input_feature_name": "C4",
            "output_feature_name": "C4_category",
            "parameters": {
                "num_buckets": 3000000
            }
        },
        {
            "ftype": tf.feature_column.categorical_column_with_identity,
            "input_feature_name": "C5",
            "output_feature_name": "C5_category",
            "parameters": {
                "num_buckets": 3000000
            }
        },
        {
            "ftype": tf.feature_column.categorical_column_with_identity,
            "input_feature_name": "C6",
            "output_feature_name": "C6_category",
            "parameters": {
                "num_buckets": 3000000
            }
        },
        {
            "ftype": tf.feature_column.categorical_column_with_identity,
            "input_feature_name": "C7",
            "output_feature_name": "C7_category",
            "parameters": {
                "num_buckets": 3000000
            }
        },
        {
            "ftype": tf.feature_column.categorical_column_with_identity,
            "input_feature_name": "C8",
            "output_feature_name": "C8_category",
            "parameters": {
                "num_buckets": 3000000
            }
        },
        {
            "ftype": tf.feature_column.categorical_column_with_identity,
            "input_feature_name": "C9",
            "output_feature_name": "C9_category",
            "parameters": {
                "num_buckets": 3000000
            }
        },
        {
            "ftype": tf.feature_column.categorical_column_with_identity,
            "input_feature_name": "C10",
            "output_feature_name": "C10_category",
            "parameters": {
                "num_buckets": 3000000
            }
        },
        {
            "ftype": tf.feature_column.categorical_column_with_identity,
            "input_feature_name": "C11",
            "output_feature_name": "C11_category",
            "parameters": {
                "num_buckets": 3000000
            }
        },
        {
            "ftype": tf.feature_column.categorical_column_with_identity,
            "input_feature_name": "C12",
            "output_feature_name": "C12_category",
            "parameters": {
                "num_buckets": 3000000
            }
        },
        {
            "ftype": tf.feature_column.categorical_column_with_identity,
            "input_feature_name": "C13",
            "output_feature_name": "C13_category",
            "parameters": {
                "num_buckets": 3000000
            }
        },
        {
            "ftype": tf.feature_column.categorical_column_with_identity,
            "input_feature_name": "C14",
            "output_feature_name": "C14_category",
            "parameters": {
                "num_buckets": 3000000
            }
        },
        {
            "ftype": tf.feature_column.categorical_column_with_identity,
            "input_feature_name": "C15",
            "output_feature_name": "C15_category",
            "parameters": {
                "num_buckets": 3000000
            }
        },
        {
            "ftype": tf.feature_column.categorical_column_with_identity,
            "input_feature_name": "C16",
            "output_feature_name": "C16_category",
            "parameters": {
                "num_buckets": 3000000
            }
        },
        {
            "ftype": tf.feature_column.categorical_column_with_identity,
            "input_feature_name": "C17",
            "output_feature_name": "C17_category",
            "parameters": {
                "num_buckets": 3000000
            }
        },
        {
            "ftype": tf.feature_column.categorical_column_with_identity,
            "input_feature_name": "C18",
            "output_feature_name": "C18_category",
            "parameters": {
                "num_buckets": 3000000
            }
        },
        {
            "ftype": tf.feature_column.categorical_column_with_identity,
            "input_feature_name": "C19",
            "output_feature_name": "C19_category",
            "parameters": {
                "num_buckets": 3000000
            }
        },
        {
            "ftype": tf.feature_column.categorical_column_with_identity,
            "input_feature_name": "C20",
            "output_feature_name": "C20_category",
            "parameters": {
                "num_buckets": 3000000
            }
        },
        {
            "ftype": tf.feature_column.categorical_column_with_identity,
            "input_feature_name": "C21",
            "output_feature_name": "C21_category",
            "parameters": {
                "num_buckets": 3000000
            }
        },
        {
            "ftype": tf.feature_column.categorical_column_with_identity,
            "input_feature_name": "C22",
            "output_feature_name": "C22_category",
            "parameters": {
                "num_buckets": 3000000
            }
        },
        {
            "ftype": tf.feature_column.categorical_column_with_identity,
            "input_feature_name": "C23",
            "output_feature_name": "C23_category",
            "parameters": {
                "num_buckets": 3000000
            }
        },
        {
            "ftype": tf.feature_column.categorical_column_with_identity,
            "input_feature_name": "C24",
            "output_feature_name": "C24_category",
            "parameters": {
                "num_buckets": 3000000
            }
        },
        {
            "ftype": tf.feature_column.categorical_column_with_identity,
            "input_feature_name": "C25",
            "output_feature_name": "C25_category",
            "parameters": {
                "num_buckets": 3000000
            }
        },
        {
            "ftype": tf.feature_column.categorical_column_with_identity,
            "input_feature_name": "C26",
            "output_feature_name": "C26_category",
            "parameters": {
                "num_buckets": 3000000
            }
        },
        {
            "ftype": tf.feature_column.embedding_column,
            "input_feature_name": "C1_category",
            "output_feature_name": "C1_category_embedding",
            "parameters": {
                "dimension": 16,
                "combiner": "sum"
            }
        },
        {
            "ftype": tf.feature_column.embedding_column,
            "input_feature_name": "C2_category",
            "output_feature_name": "C2_category_embedding",
            "parameters": {
                "dimension": 16,
                "combiner": "sum"
            }
        },
        {
            "ftype": tf.feature_column.embedding_column,
            "input_feature_name": "C3_category",
            "output_feature_name": "C3_category_embedding",
            "parameters": {
                "dimension": 16,
                "combiner": "sum"
            }
        },
        {
            "ftype": tf.feature_column.embedding_column,
            "input_feature_name": "C4_category",
            "output_feature_name": "C4_category_embedding",
            "parameters": {
                "dimension": 16,
                "combiner": "sum"
            }
        },
        {
            "ftype": tf.feature_column.embedding_column,
            "input_feature_name": "C5_category",
            "output_feature_name": "C5_category_embedding",
            "parameters": {
                "dimension": 16,
                "combiner": "sum"
            }
        },
        {
            "ftype": tf.feature_column.embedding_column,
            "input_feature_name": "C6_category",
            "output_feature_name": "C6_category_embedding",
            "parameters": {
                "dimension": 16,
                "combiner": "sum"
            }
        },
        {
            "ftype": tf.feature_column.embedding_column,
            "input_feature_name": "C7_category",
            "output_feature_name": "C7_category_embedding",
            "parameters": {
                "dimension": 16,
                "combiner": "sum"
            }
        },
        {
            "ftype": tf.feature_column.embedding_column,
            "input_feature_name": "C8_category",
            "output_feature_name": "C8_category_embedding",
            "parameters": {
                "dimension": 16,
                "combiner": "sum"
            }
        },
        {
            "ftype": tf.feature_column.embedding_column,
            "input_feature_name": "C9_category",
            "output_feature_name": "C9_category_embedding",
            "parameters": {
                "dimension": 16,
                "combiner": "sum"
            }
        },
        {
            "ftype": tf.feature_column.embedding_column,
            "input_feature_name": "C10_category",
            "output_feature_name": "C10_category_embedding",
            "parameters": {
                "dimension": 16,
                "combiner": "sum"
            }
        },
        {
            "ftype": tf.feature_column.embedding_column,
            "input_feature_name": "C11_category",
            "output_feature_name": "C11_category_embedding",
            "parameters": {
                "dimension": 16,
                "combiner": "sum"
            }
        },
        {
            "ftype": tf.feature_column.embedding_column,
            "input_feature_name": "C12_category",
            "output_feature_name": "C12_category_embedding",
            "parameters": {
                "dimension": 16,
                "combiner": "sum"
            }
        },
        {
            "ftype": tf.feature_column.embedding_column,
            "input_feature_name": "C13_category",
            "output_feature_name": "C13_category_embedding",
            "parameters": {
                "dimension": 16,
                "combiner": "sum"
            }
        },
        {
            "ftype": tf.feature_column.embedding_column,
            "input_feature_name": "C14_category",
            "output_feature_name": "C14_category_embedding",
            "parameters": {
                "dimension": 16,
                "combiner": "sum"
            }
        },
        {
            "ftype": tf.feature_column.embedding_column,
            "input_feature_name": "C15_category",
            "output_feature_name": "C15_category_embedding",
            "parameters": {
                "dimension": 16,
                "combiner": "sum"
            }
        },
        {
            "ftype": tf.feature_column.embedding_column,
            "input_feature_name": "C16_category",
            "output_feature_name": "C16_category_embedding",
            "parameters": {
                "dimension": 16,
                "combiner": "sum"
            }
        },
        {
            "ftype": tf.feature_column.embedding_column,
            "input_feature_name": "C17_category",
            "output_feature_name": "C17_category_embedding",
            "parameters": {
                "dimension": 16,
                "combiner": "sum"
            }
        },
        {
            "ftype": tf.feature_column.embedding_column,
            "input_feature_name": "C18_category",
            "output_feature_name": "C18_category_embedding",
            "parameters": {
                "dimension": 16,
                "combiner": "sum"
            }
        },
        {
            "ftype": tf.feature_column.embedding_column,
            "input_feature_name": "C19_category",
            "output_feature_name": "C19_category_embedding",
            "parameters": {
                "dimension": 16,
                "combiner": "sum"
            }
        },
        {
            "ftype": tf.feature_column.embedding_column,
            "input_feature_name": "C20_category",
            "output_feature_name": "C20_category_embedding",
            "parameters": {
                "dimension": 16,
                "combiner": "sum"
            }
        },
        {
            "ftype": tf.feature_column.embedding_column,
            "input_feature_name": "C21_category",
            "output_feature_name": "C21_category_embedding",
            "parameters": {
                "dimension": 16,
                "combiner": "sum"
            }
        },
        {
            "ftype": tf.feature_column.embedding_column,
            "input_feature_name": "C22_category",
            "output_feature_name": "C22_category_embedding",
            "parameters": {
                "dimension": 16,
                "combiner": "sum"
            }
        },
        {
            "ftype": tf.feature_column.embedding_column,
            "input_feature_name": "C23_category",
            "output_feature_name": "C23_category_embedding",
            "parameters": {
                "dimension": 16,
                "combiner": "sum"
            }
        },
        {
            "ftype": tf.feature_column.embedding_column,
            "input_feature_name": "C24_category",
            "output_feature_name": "C24_category_embedding",
            "parameters": {
                "dimension": 16,
                "combiner": "sum"
            }
        },
        {
            "ftype": tf.feature_column.embedding_column,
            "input_feature_name": "C25_category",
            "output_feature_name": "C25_category_embedding",
            "parameters": {
                "dimension": 16,
                "combiner": "sum"
            }
        },
        {
            "ftype": tf.feature_column.embedding_column,
            "input_feature_name": "C26_category",
            "output_feature_name": "C26_category_embedding",
            "parameters": {
                "dimension": 16,
                "combiner": "sum"
            }
        },
        {
            "ftype": tf.feature_column.embedding_column,
            "input_feature_name": "I1_numeral_bucket",
            "output_feature_name": "I1_numeral_bucket_embedding",
            "parameters": {
                "dimension": 16,
                "combiner": "sum"
            }
        },
        {
            "ftype": tf.feature_column.embedding_column,
            "input_feature_name": "I2_numeral_bucket",
            "output_feature_name": "I2_numeral_bucket_embedding",
            "parameters": {
                "dimension": 16,
                "combiner": "sum"
            }
        },
        {
            "ftype": tf.feature_column.embedding_column,
            "input_feature_name": "I3_numeral_bucket",
            "output_feature_name": "I3_numeral_bucket_embedding",
            "parameters": {
                "dimension": 16,
                "combiner": "sum"
            }
        },
        {
            "ftype": tf.feature_column.embedding_column,
            "input_feature_name": "I4_numeral_bucket",
            "output_feature_name": "I4_numeral_bucket_embedding",
            "parameters": {
                "dimension": 16,
                "combiner": "sum"
            }
        },
        {
            "ftype": tf.feature_column.embedding_column,
            "input_feature_name": "I5_numeral_bucket",
            "output_feature_name": "I5_numeral_bucket_embedding",
            "parameters": {
                "dimension": 16,
                "combiner": "sum"
            }
        },
        {
            "ftype": tf.feature_column.embedding_column,
            "input_feature_name": "I6_numeral_bucket",
            "output_feature_name": "I6_numeral_bucket_embedding",
            "parameters": {
                "dimension": 16,
                "combiner": "sum"
            }
        },
        {
            "ftype": tf.feature_column.embedding_column,
            "input_feature_name": "I7_numeral_bucket",
            "output_feature_name": "I7_numeral_bucket_embedding",
            "parameters": {
                "dimension": 16,
                "combiner": "sum"
            }
        },
        {
            "ftype": tf.feature_column.embedding_column,
            "input_feature_name": "I8_numeral_bucket",
            "output_feature_name": "I8_numeral_bucket_embedding",
            "parameters": {
                "dimension": 16,
                "combiner": "sum"
            }
        },
        {
            "ftype": tf.feature_column.embedding_column,
            "input_feature_name": "I9_numeral_bucket",
            "output_feature_name": "I9_numeral_bucket_embedding",
            "parameters": {
                "dimension": 16,
                "combiner": "sum"
            }
        },
        {
            "ftype": tf.feature_column.embedding_column,
            "input_feature_name": "I10_numeral_bucket",
            "output_feature_name": "I10_numeral_bucket_embedding",
            "parameters": {
                "dimension": 16,
                "combiner": "sum"
            }
        },
        {
            "ftype": tf.feature_column.embedding_column,
            "input_feature_name": "I11_numeral_bucket",
            "output_feature_name": "I11_numeral_bucket_embedding",
            "parameters": {
                "dimension": 16,
                "combiner": "sum"
            }
        },
        {
            "ftype": tf.feature_column.embedding_column,
            "input_feature_name": "I12_numeral_bucket",
            "output_feature_name": "I12_numeral_bucket_embedding",
            "parameters": {
                "dimension": 16,
                "combiner": "sum"
            }
        },
        {
            "ftype": tf.feature_column.embedding_column,
            "input_feature_name": "I13_numeral_bucket",
            "output_feature_name": "I13_numeral_bucket_embedding",
            "parameters": {
                "dimension": 16,
                "combiner": "sum"
            }
        }
    ],
    "feature_column_group": {
        "wide": [
            "C1_category",
            "C2_category",
            "C3_category",
            "C4_category",
            "C5_category",
            "C6_category",
            "C7_category",
            "C8_category",
            "C9_category",
            "C10_category",
            "C11_category",
            "C12_category",
            "C13_category",
            "C14_category",
            "C15_category",
            "C16_category",
            "C17_category",
            "C18_category",
            "C19_category",
            "C20_category",
            "C21_category",
            "C22_category",
            "C23_category",
            "C24_category",
            "C25_category",
            "C26_category"
        ],
        "deep": [
            "C1_category_embedding",
            "C2_category_embedding",
            "C3_category_embedding",
            "C4_category_embedding",
            "C5_category_embedding",
            "C6_category_embedding",
            "C7_category_embedding",
            "C8_category_embedding",
            "C9_category_embedding",
            "C10_category_embedding",
            "C11_category_embedding",
            "C12_category_embedding",
            "C13_category_embedding",
            "C14_category_embedding",
            "C15_category_embedding",
            "C16_category_embedding",
            "C17_category_embedding",
            "C18_category_embedding",
            "C19_category_embedding",
            "C20_category_embedding",
            "C21_category_embedding",
            "C22_category_embedding",
            "C23_category_embedding",
            "C24_category_embedding",
            "C25_category_embedding",
            "C26_category_embedding",
            "I1_numeral_bucket_embedding",
            "I2_numeral_bucket_embedding",
            "I3_numeral_bucket_embedding",
            "I4_numeral_bucket_embedding",
            "I5_numeral_bucket_embedding",
            "I6_numeral_bucket_embedding",
            "I7_numeral_bucket_embedding",
            "I8_numeral_bucket_embedding",
            "I9_numeral_bucket_embedding",
            "I10_numeral_bucket_embedding",
            "I11_numeral_bucket_embedding",
            "I12_numeral_bucket_embedding",
            "I13_numeral_bucket_embedding"
        ],
        "cross": [
            "C1_category_embedding",
            "C2_category_embedding",
            "C3_category_embedding",
            "C4_category_embedding",
            "C5_category_embedding",
            "C6_category_embedding",
            "C7_category_embedding",
            "C8_category_embedding",
            "C9_category_embedding",
            "C10_category_embedding",
            "C11_category_embedding",
            "C12_category_embedding",
            "C13_category_embedding",
            "C14_category_embedding",
            "C15_category_embedding",
            "C16_category_embedding",
            "C17_category_embedding",
            "C18_category_embedding",
            "C19_category_embedding",
            "C20_category_embedding",
            "C21_category_embedding",
            "C22_category_embedding",
            "C23_category_embedding",
            "C24_category_embedding",
            "C25_category_embedding",
            "C26_category_embedding",
            "I1_numeral_bucket_embedding",
            "I2_numeral_bucket_embedding",
            "I3_numeral_bucket_embedding",
            "I4_numeral_bucket_embedding",
            "I5_numeral_bucket_embedding",
            "I6_numeral_bucket_embedding",
            "I7_numeral_bucket_embedding",
            "I8_numeral_bucket_embedding",
            "I9_numeral_bucket_embedding",
            "I10_numeral_bucket_embedding",
            "I11_numeral_bucket_embedding",
            "I12_numeral_bucket_embedding",
            "I13_numeral_bucket_embedding"
        ],
        "fm": [
            "C1_category_embedding",
            "C2_category_embedding",
            "C3_category_embedding",
            "C4_category_embedding",
            "C5_category_embedding",
            "C6_category_embedding",
            "C7_category_embedding",
            "C8_category_embedding",
            "C9_category_embedding",
            "C10_category_embedding",
            "C11_category_embedding",
            "C12_category_embedding",
            "C13_category_embedding",
            "C14_category_embedding",
            "C15_category_embedding",
            "C16_category_embedding",
            "C17_category_embedding",
            "C18_category_embedding",
            "C19_category_embedding",
            "C20_category_embedding",
            "C21_category_embedding",
            "C22_category_embedding",
            "C23_category_embedding",
            "C24_category_embedding",
            "C25_category_embedding",
            "C26_category_embedding",
            "I1_numeral_bucket_embedding",
            "I2_numeral_bucket_embedding",
            "I3_numeral_bucket_embedding",
            "I4_numeral_bucket_embedding",
            "I5_numeral_bucket_embedding",
            "I6_numeral_bucket_embedding",
            "I7_numeral_bucket_embedding",
            "I8_numeral_bucket_embedding",
            "I9_numeral_bucket_embedding",
            "I10_numeral_bucket_embedding",
            "I11_numeral_bucket_embedding",
            "I12_numeral_bucket_embedding",
            "I13_numeral_bucket_embedding"
        ],
        "attention_1": [],
        "attention_2": [],
        "attention_3": [],
        "attention_4": [],
        "pnn_1": [],
        "pnn_2": [],
        "pnn_3": [],
        "pnn_4": []
    }
}
