#!/usr/bin/python
# -*- encoding: utf-8 -*-
# @Author: gavinlzhang
# @Date: 2022/08/08
import sys
from model.model_wdcc_point import *
from model.estimator import *
from feature_config.model_json_conf import FEATURE_CONFIG
from feature_config.model_input_fn import FEATURE_TRANSFORM_CONFIG
from utils.common import *

FLAGS = tf.app.flags.FLAGS

# run mode
tf.app.flags.DEFINE_enum("run_mode", "train", ["train_and_evaluate", "train", "predict", "eval", "export"], "Run Mode")

# input and output para
tf.app.flags.DEFINE_string("train_data_dir", "./train_data_dir", "Dir For Training Data")
tf.app.flags.DEFINE_string("eval_data_dir", "./eval_data_dir", "Dir For Evaluation Data")
tf.app.flags.DEFINE_string("predict_data_dir", "./predict_data_dir", "Predict Input Data")
tf.app.flags.DEFINE_string("predict_data_output_dir", "./predict_data_output_dir", "Output Predict Data")
tf.app.flags.DEFINE_string("export_dir", "./export_dir", "Path For Export PB Models")
tf.app.flags.DEFINE_string("export_signature_def", "banner", "Output Signature Def")
tf.app.flags.DEFINE_boolean("export_str_input", True, "Give Export Model Str Input")
tf.app.flags.DEFINE_enum("dataset_compression_types", "GZIP", ["", "ZLIB", "GZIP"], "Data Set Compression Type")

# train para
tf.app.flags.DEFINE_integer("batch_size", 128, "Training Batch Size")
tf.app.flags.DEFINE_integer("num_epochs", 5, "Training Num Epochs")
# train para
tf.app.flags.DEFINE_float("deep_lr", 0.001, "Learning rate of DNN")
tf.app.flags.DEFINE_float("deep_l1", 0.01, "L1 reg of DNN")
tf.app.flags.DEFINE_float("deep_l2", 0.005, "L2 reg of DNN")
tf.app.flags.DEFINE_float("wide_lr", 0.001, "Learning rate of Wide")
tf.app.flags.DEFINE_float("wide_l1", 0.3, "L1 reg of Wide")
tf.app.flags.DEFINE_float("wide_l2", 0.05, "L2 reg of Wide")

# estimator para
tf.app.flags.DEFINE_string("model_dir", "./model_dir", "Path for Export PB Models")
tf.app.flags.DEFINE_integer("tf_random_seed", 1024, "Training Random Seed")
tf.app.flags.DEFINE_integer("save_summary_steps", 100, "Summary Per N Steps")
tf.app.flags.DEFINE_integer("save_checkpoints_steps", 5000, "Save CKPT Per N Steps")
tf.app.flags.DEFINE_integer("keep_checkpoint_max", 3, "Max CKPT Keeped")
tf.app.flags.DEFINE_integer("keep_checkpoint_every_n_hours", 10000, "Keep CKPT Every N Hour")
tf.app.flags.DEFINE_integer("log_step_count_steps", 100, "Print Log Per N Steps")

# machine source control
tf.app.flags.DEFINE_integer("device_cpu_count", 8, "Use CPU Count")
tf.app.flags.DEFINE_integer("device_gpu_count", 8, "Use GPU Count")
tf.app.flags.DEFINE_integer("inter_op_parallelism_threads", 8, "Inter OP Cal Threads")
tf.app.flags.DEFINE_integer("intra_op_parallelism_threads", 8, "Intra OP Cal Threads")

# distribution para
tf.app.flags.DEFINE_string("ps_hosts", "", "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", "", "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")

def get_run_config():
    config = {
        'model_dir': FLAGS.model_dir,
        'save_summary_steps': FLAGS.save_summary_steps,
        'save_checkpoints_steps': FLAGS.save_checkpoints_steps,
        'keep_checkpoint_max': FLAGS.keep_checkpoint_max,
        'keep_checkpoint_every_n_hours': FLAGS.keep_checkpoint_every_n_hours,
        'log_step_count_steps': FLAGS.log_step_count_steps,
    }
    if FLAGS.tf_random_seed > 0:
        config['tf_random_seed'] = FLAGS.tf_random_seed
    device_count = {}
    if FLAGS.device_cpu_count > 0:
        device_count["CPU"] = FLAGS.device_cpu_count
    if FLAGS.device_gpu_count > 0:
        device_count["GPU"] = FLAGS.device_gpu_count
    config_proto = tf.ConfigProto(
        device_count=device_count,
        log_device_placement=False,
        allow_soft_placement=True,
        inter_op_parallelism_threads=FLAGS.inter_op_parallelism_threads,
        intra_op_parallelism_threads=FLAGS.intra_op_parallelism_threads)
    config_proto.gpu_options.allow_growth = True
    config['session_config'] = config_proto
    return tf.estimator.RunConfig(**config)

def get_current_step():
    step = 0
    try:
        files = os.listdir(FLAGS.model_dir)
        for one in files:
            if "model.ckpt" in one:
                value = int(one.split(".")[1].split("-")[1])
                if value > step:
                    step = value
    except:
        pass
    return step

def main(para):
    batch_size = FLAGS.batch_size
    num_epochs = FLAGS.num_epochs
    run_config = get_run_config()
    file_generator = FileNameIterator(run_config)
    tf.logging.info("RunConfig:%s", run_config)
    tf.logging.info("BatchSize:%d NumEpochs:%d ", batch_size, num_epochs)
    tf.logging.info("CurrentStep:%d", get_current_step())
    data_input = DataInput(FEATURE_CONFIG)
    data_transformer = FeatureTransformer(FEATURE_TRANSFORM_CONFIG)
    feature_transformed_groups = data_transformer.get_model_features_groups()
    model_params = {
        'wide_feature_columns': feature_transformed_groups['wide'],
        'deep_feature_columns': feature_transformed_groups['deep'],
        'cross_feature_columns': feature_transformed_groups['cross'],
        'compressed_feature_columns': None,
        'fm_feature_columns': feature_transformed_groups['fm'],
        'fm_embedding': 16,
        'attention_list': [
            feature_transformed_groups['attention_1'],
            feature_transformed_groups['attention_2'],
            feature_transformed_groups['attention_3'],
            feature_transformed_groups['attention_4'],
        ],
        'attention_hidden_units': [64],
        'pnn_list': [
            feature_transformed_groups['pnn_1'],
            feature_transformed_groups['pnn_2'],
            feature_transformed_groups['pnn_3'],
            feature_transformed_groups['pnn_4'],
        ],
        'output_dim': 1,
        'use_wide': True,
        'use_cross': True,
        'use_deep': True,
        'use_compressed': False,
        'use_fm': False,
        'use_attention': False,
        'use_pnn': False,
        'dcn_parallel': True,
        'deep_use_bn': False,
        'cross_use_bn': False,
        'compressed_use_bn': False,
        'deep_optimizer': tf.train.ProximalAdagradOptimizer(
            learning_rate=FLAGS.deep_lr,
            l1_regularization_strength=FLAGS.deep_l1,
            l2_regularization_strength=FLAGS.deep_l2),
        'wide_optimizer': tf.train.FtrlOptimizer(
            learning_rate=FLAGS.wide_lr,
            l1_regularization_strength=FLAGS.wide_l1,
            l2_regularization_strength=FLAGS.wide_l2),
        'use_focal_loss': False,
        'deep_dropout': 0.15,
        'deep_activate_fn': tf.nn.relu,
        'deep_hidden_units': [512, 128, 32],
        'cross_dropout': 0.10,
        'cross_layers': 2,
        'cross_l2': 0.05,
        'compressed_dropout': 0.10,
        'compressed_layers': [4],
        'compressed_embedding': 32,
        'compressed_file_split_half': True,
        'compressed_l2': 0.05
    }

    # [self.pos_wide_pre, self.pos_deep_pre, self.pre]
    model_fn_params = ModelFnParams(
        WDCCNetwork(**model_params), ["wide_pre", "deep_pre", "all_pre"], ["C1", "C2"],
        FLAGS.export_signature_def
    )
    tf.logging.info("ModelConfig:%s", model_params)
    model = get_custom_estimator(model_fn_params, run_config)
    com = FLAGS.dataset_compression_types
    predict_input_fn = lambda x: data_input.get_dataset_from_csv_v3(x, 1, False, batch_size, False, com)
    train_input_fn = lambda x: data_input.get_dataset_from_csv_v3(x, num_epochs, True, batch_size, True, com)
    eval_input_fn = lambda x: data_input.get_dataset_from_csv_v3(x, num_epochs, False, batch_size, True, com)

    if FLAGS.run_mode == "train":
        train_path = FLAGS.train_data_dir
        train_files = file_generator.get_file_names([train_path])
        random.shuffle(train_files)
        tf.logging.info("TrainFilesNum=%d TrainPath=%s", len(train_files), train_path)
        train_data = lambda: train_input_fn(train_files)
        model.train(train_data, hooks=[])
        tf.logging.info("Train End")

    if FLAGS.run_mode == "eval":
        eval_path = FLAGS.eval_data_dir
        eval_files = file_generator.get_file_names([eval_path])
        tf.logging.info("EvalFilesNum=%d EvalPath=%s", len(eval_files), eval_path)
        eval_data = lambda: eval_input_fn(eval_files)
        results = model.evaluate(eval_data)
        tf.logging.info("Eval End Result=%s", results)
        file_generator.write_to_file_list_append("xxxx.eval", results)

    if FLAGS.run_mode == "predict":
        predict_path = FLAGS.predict_data_dir
        output_predict_path = FLAGS.predict_data_output_dir
        predict_files = file_generator.get_file_names([predict_path])
        tf.logging.info("PredictFilesNum=%d PredictPath=%s", len(predict_files), predict_path)
        predict_data = lambda: predict_input_fn(predict_files)
        results = []
        for result in model.predict(predict_data):
            results.append(result)
            # break
        file_generator.write_to_file(output_predict_path, results)
        tf.logging.info("Predict End ResultNum=%d Path=%s", len(results), output_predict_path)

    if FLAGS.run_mode == "export":
        export_dir = FLAGS.export_dir
        if not FLAGS.export_str_input:
            serving_input = data_input.serving_input_fn_from_feed
        else:
            serving_input = data_input.serving_input_fn_from_string
        model.export_savedmodel(export_dir_base=export_dir, serving_input_receiver_fn=serving_input)

    if FLAGS.run_mode == "train_and_evaluate":
        train_path = FLAGS.train_data_dir
        train_files = file_generator.get_file_names([train_path])
        random.shuffle(train_files)
        eval_path = FLAGS.eval_data_dir
        eval_files = file_generator.get_file_names([eval_path])
        tf.logging.info("TrainFilesNum=%d TrainPath=%s", len(train_files), train_path)
        tf.logging.info("EvalFilesNum=%d EvalPath=%s", len(eval_files), eval_path)
        train_spec = tf.estimator.TrainSpec(input_fn=lambda: train_input_fn(train_files), hooks=[])
        eval_spec = tf.estimator.EvalSpec(input_fn=lambda: eval_input_fn(eval_files), steps=None)
        tf.estimator.train_and_evaluate(model, train_spec, eval_spec)

if __name__ == "__main__":
    tf.app.run()
