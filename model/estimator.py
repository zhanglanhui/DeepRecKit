# -*- encoding:utf-8 -*-

import json
import os
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.estimator.export.export import build_all_signature_defs
from tensorflow.python.estimator.export.export import get_temp_export_dir
from tensorflow.python.estimator.export.export import get_timestamped_export_dir
from tensorflow.python.estimator import model_fn as model_fn_lib
from tensorflow.python.training import saver
from tensorflow.python.client import session as tf_session
from tensorflow.python.platform import gfile
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.util import compat
from tensorflow.python.ops import resources
from tensorflow.python.ops import variables
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import lookup_ops

class ModelBase:
    def __init__(self, model_name):
        self.model_name = model_name
        tf.logging.info("Model Base Init %s", self.model_name)
        self.pre = None
        self.loss = None
        self.optimizer = None
        self.train_op = None
        self.middle_output = {}

    def build_graph(self, features, targets, mode, config):
        pass

    def get_pre(self):
        return self.pre

    def get_middle_output(self):
        return self.middle_output

    def get_loss(self):
        return self.loss

    def get_optimizer(self):
        return self.optimizer

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def get_optimizers(self):
        return []

    def set_optimizers(self, optimizers):
        pass

    def get_train_op(self):
        return self.train_op

    def get_metrics(self, targets, preds):
        return {}

    def get_eval_summary(self, targets, preds):
        return {}

    @staticmethod
    def build_deep_layers(net, hidden_units, activate_fn, dropout=0.0):
        for num_hidden_units in hidden_units:
            net = tf.layers.dense(
                net,
                units=num_hidden_units,
                activation=activate_fn,
                kernel_initializer=tf.glorot_uniform_initializer())
            if 1.0 > dropout > 0.0:
                net = tf.nn.dropout(net, keep_prob=1 - dropout)
        return net

class ModelFnParams:
    def __init__(self, model,
                 predict_output_name="targets",
                 predict_feature_output_names=[],
                 export_signature_def="serving_default",
                 sync=False,
                 training_chief_hooks=None,
                 training_hooks=None,
                 evaluation_hooks=None,
                 prediction_hooks=None):
        if not isinstance(model, ModelBase):
            raise Exception("Model Base Needed")
        if not isinstance(predict_feature_output_names, list):
            raise Exception("Predict Feature Output Names Must Be List")
        self.__predict_output_name = predict_output_name
        self.__predict_feature_output_names = predict_feature_output_names
        self.__export_signature_def = export_signature_def
        self.__model = model
        self.__sync = sync
        self.__training_chief_hooks = training_chief_hooks
        self.__training_hooks = training_hooks
        self.__evaluation_hooks = evaluation_hooks
        self.__prediction_hooks = prediction_hooks

    def get_predict_output_name(self):
        return self.__predict_output_name

    def get_predict_feature_output_names(self):
        return self.__predict_feature_output_names

    def get_export_signature_def(self):
        return self.__export_signature_def

    def get_model(self):
        return self.__model

    def is_sync(self):
        return self.__sync

    def get_training_hooks(self):
        hooks = []
        if self.__training_hooks is not None:
            for i in self.__training_hooks:
                hooks.append(i)
        return hooks

    def get_evaluation_hooks(self):
        hooks = []
        if self.__evaluation_hooks is not None:
            for i in self.__evaluation_hooks:
                hooks.append(i)
        return hooks

    def get_training_chief_hooks(self):
        hooks = []
        if self.__training_chief_hooks is not None:
            for i in self.__training_chief_hooks:
                hooks.append(i)
        return hooks

    def get_prediction_hooks(self):
        hooks = []
        if self.__prediction_hooks is not None:
            for i in self.__prediction_hooks:
                hooks.append(i)
        return hooks

    def __str__(self):
        return json.dumps({
            "models": self.__model.model_name,
            "predict_output_name": self.__predict_output_name,
            "predict_feature_output_names": self.__predict_feature_output_names,
            "export_signature_def": self.__export_signature_def,
            "sync": self.__sync})

def export_model(estimator,
                 export_dir_base,
                 serving_input_receiver_fn,
                 assets_extra=None,
                 as_text=False,
                 checkpoint_path=None,
                 strip_default_attrs=False):
    if serving_input_receiver_fn is None:
        raise ValueError('Serving Input Receiver Fn Can Not Be Defined.')
    if not isinstance(estimator, tf.estimator.Estimator):
        raise ValueError("Estimator Needed")
    with ops.Graph().as_default() as g:
        estimator._create_and_assert_global_step(g)
        random_seed.set_random_seed(estimator._config.tf_random_seed)
        serving_input_receiver = serving_input_receiver_fn()
        estimator_spec = estimator._call_model_fn(
            features=serving_input_receiver.features,
            labels=None,
            mode=model_fn_lib.ModeKeys.PREDICT,
            config=estimator.config)
        signature_def_map = build_all_signature_defs(
            serving_input_receiver.receiver_tensors,
            estimator_spec.export_outputs,
            serving_input_receiver.receiver_tensors_alternatives)
        if not checkpoint_path:
            checkpoint_path = saver.latest_checkpoint(estimator._model_dir)
        if not checkpoint_path:
            raise ValueError("Couldn't find trained model at %s." % estimator._model_dir)
        export_dir = get_timestamped_export_dir(export_dir_base)
        temp_export_dir = get_temp_export_dir(export_dir)
        # TODO(soergel): Consider whether MonitoredSession makes sense here
        with tf_session.Session(config=estimator._session_config) as session:
            saveables = variables._all_saveable_objects()
            if (estimator_spec.scaffold is not None and
                    estimator_spec.scaffold.saver is not None):
                saver_for_restore = estimator_spec.scaffold.saver
            elif saveables:
                saver_for_restore = saver.Saver(saveables, sharded=True)
            saver_for_restore.restore(session, checkpoint_path)

            # TODO(b/36111876): replace legacy_init_op with main_op mechanism
            init_op = control_flow_ops.group(variables.local_variables_initializer(),
                                             resources.initialize_resources(resources.shared_resources()),
                                             lookup_ops.tables_initializer())
            builder = saved_model_builder.SavedModelBuilder(temp_export_dir)
            builder.add_meta_graph_and_variables(
                session, [tag_constants.SERVING],
                signature_def_map=signature_def_map,
                assets_collection=ops.get_collection(ops.GraphKeys.ASSET_FILEPATHS),
                main_op=init_op,
                strip_default_attrs=strip_default_attrs)
            builder.save(as_text)
    if assets_extra:
        assets_extra_path = os.path.join(
            compat.as_bytes(temp_export_dir), compat.as_bytes('assets.extra'))
        for dest_relative, source in assets_extra.items():
            dest_absolute = os.path.join(compat.as_bytes(assets_extra_path),
                                         compat.as_bytes(dest_relative))
            dest_path = os.path.dirname(dest_absolute)
            gfile.MakeDirs(dest_path)
            gfile.Copy(source, dest_absolute)
    gfile.Rename(temp_export_dir, export_dir)
    return export_dir

def standard_model_fn(features, labels, mode, params, config):
    if not isinstance(params, ModelFnParams):
        raise Exception("ModelFnParams Needed")
    tf.logging.info("Mode=%s Params=%s Session Config=%s", mode, params, config.session_config)
    label = labels
    model = params.get_model()
    training_hooks = params.get_training_hooks()
    training_chief_hooks = params.get_training_chief_hooks()
    evaluation_hooks = params.get_evaluation_hooks()
    prediction_hooks = params.get_prediction_hooks()
    model.build_graph(features, label, mode, config)
    preds = model.get_pre()
    loss = model.get_loss()
    optimizer = model.get_optimizer()
    if mode == tf.estimator.ModeKeys.TRAIN:
        summary = model.get_eval_summary(labels, preds)
        for key, value in summary.items():
            tf.summary.scalar(key, value)
    if mode == tf.estimator.ModeKeys.EVAL:
        metrics = model.get_metrics(label, preds)
        return tf.estimator.EstimatorSpec(mode,
                                          loss=loss,
                                          eval_metric_ops=metrics,
                                          evaluation_hooks=evaluation_hooks)
    if mode == tf.estimator.ModeKeys.PREDICT:
        if isinstance(preds, list):
            predictions = dict(zip(params.get_predict_output_name(), preds))
        else:
            predictions = {params.get_predict_output_name(): preds}
        for feature_output_name in params.get_predict_feature_output_names():
            predictions[feature_output_name] = tf.identity(features[feature_output_name], name=feature_output_name)
        middle_output = model.get_middle_output()
        if middle_output is not None and len(middle_output) > 0:
            predictions.update(middle_output)
        export_outputs = {
            params.get_export_signature_def(): tf.estimator.export.PredictOutput(predictions)
        }
        return tf.estimator.EstimatorSpec(mode,
                                          predictions=predictions,
                                          export_outputs=export_outputs,
                                          prediction_hooks=prediction_hooks)
    if model.get_train_op() is None:
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    else:
        train_op = model.get_train_op()
    # training_hooks = [tf.train.ProfilerHook(save_steps=100, output_dir='/data/ceph/minister/profile')]
    spec = tf.estimator.EstimatorSpec(mode,
                                      loss=loss,
                                      train_op=train_op,
                                      training_hooks=training_hooks,
                                      training_chief_hooks=training_chief_hooks)
    return spec

def get_custom_estimator(model_fn_params, run_config, model_fn=standard_model_fn):
    return tf.estimator.Estimator(
        model_fn=model_fn,
        params=model_fn_params,
        config=run_config
    )

def get_average_classifier(**kwargs):
    return tf.estimator.BaselineClassifier(**kwargs)

def get_average_regressor(**kwargs):
    return tf.estimator.BaselineRegressor(**kwargs)

def get_linear_classifier(**kwargs):
    return tf.estimator.LinearClassifier(**kwargs)

def get_linear_regressor(**kwargs):
    return tf.estimator.LinearRegressor(**kwargs)

def get_dnn_classifier(**kwargs):
    return tf.estimator.DNNClassifier(**kwargs)

def get_dnn_regressor(**kwargs):
    return tf.estimator.DNNRegressor(**kwargs)

def get_wad_classifier(**kwargs):
    return tf.estimator.DNNLinearCombinedClassifier(**kwargs)

def get_wad_regressor(**kwargs):
    return tf.estimator.DNNLinearCombinedRegressor(**kwargs)

def get_boost_classifier(**kwargs):
    return tf.estimator.BoostedTreesClassifier(**kwargs)

def get_boost_regressor(**kwargs):
    return tf.estimator.BoostedTreesRegressor(**kwargs)
