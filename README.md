# DeepRecKit

DeepRecKit is a TensorFlow 1.x based recommendation training repository. The core workflow of this project is:

1. define raw feature metadata in a JSON file
2. generate TensorFlow feature configuration code from that JSON
3. load CSV data with the generated schema
4. build feature columns and feature groups
5. run model training, evaluation, prediction, or export through `model_train.py`

The repository is centered on the TensorFlow training stack under `model/`, `feature_config/`, `features/`, and `utils/`.

## Main Entry Point

The main training entry point is:

```bash
python model_train.py
```

The convenience wrapper is:

```bash
bash model_train.sh
```

## Supported Environment

This repository targets the TensorFlow 1.x API style.

Recommended environment:

- Python 3.7
- TensorFlow 1.15 compatible runtime
- numpy
- pandas

Important notes:

- The code uses `tf.app.flags`, `tf.estimator`, `tf.feature_column`, and other TensorFlow 1.x APIs.
- The main training path has not been migrated to TensorFlow 2.x.
- The repository is intended for local experimentation and reproducibility, not production deployment as-is.

## What This Repository Supports

DeepRecKit currently supports the following workflow.

## Supported Algorithms

The repository includes multiple TensorFlow recommendation model implementations under `model/`.

### Point-wise CTR / ranking style models

- `model_wnd.py`: Wide & Deep
- `model_dcn.py`: Deep & Cross Network
- `model_deepfm.py` and `deepfm.py`: DeepFM variants
- `model_wdcc_point.py`: a combined wide/deep/cross style point-wise model
- `model_combine.py`: a combined feature-column based ranking model

### Pair-wise / matching style models

- `model_wdcc_pair.py`: pair-wise wide/deep/cross style model
- `model_dssm.py`: DSSM-style two-tower matching model

### Multi-task models

- `model_mtl.py`: multi-task learning model
- `model_mtl_pair.py`: pair-wise multi-task model
- `model_mtl_mfh.py`: multi-task model with MFH-style task handling

### Feature interaction and auxiliary components

The repository also includes reusable model building blocks such as:

- FM interaction layers
- crossed features
- embedding and shared embedding transforms
- attention-related feature groups
- PNN-related feature groups
- co-action style feature interaction modules

The exact set of active feature groups depends on the generated `FEATURE_TRANSFORM_CONFIG` and the model parameters selected in `model_train.py`.

### 1. Raw feature specification in JSON

Feature metadata is defined in a JSON file such as:

```text
data/open_data/criteo_sample/criteo_sample_feature_config_bucketized.json
```

This JSON is used to describe feature properties such as:

- feature dtype
- whether a feature is a label
- embedding dimension
- bucket boundaries
- vocabulary size
- shared embedding name
- sequence length or separator information

### 2. Automatic generation of Python feature config files

Run:

```bash
bash model_config.sh
```

This script reads the raw JSON config and generates Python config files under `feature_config/`.

Generated files:

- `feature_config/model_json_conf.py`
- `feature_config/model_input_fn.py`

These files are generated from:

- `configs/config_preprocess_fields.py`
- `configs/config_train_fields.py`
- `configs/config_model_struct.py`

### 3. Data schema generation for input parsing

`feature_config/model_json_conf.py` is generated for data input parsing.

It defines:

- feature name
- shape
- `tf.FixedLenFeature` or `tf.VarLenFeature`
- dtype
- default value
- whether the field is a feature or target

This generated schema is used by `DataInput` in `utils/common.py` to read CSV input and split features from `label`.

### 4. Feature-column and feature-group generation

`feature_config/model_input_fn.py` is generated for TensorFlow feature transformation.

It contains:

- `FEATURE_CONFIG`
- `FEATURE_TRANSFORM_CONFIG`

`FEATURE_TRANSFORM_CONFIG` describes how raw fields are transformed into TensorFlow feature columns, including:

- numeric columns
- bucketized columns
- categorical identity columns
- hash bucket columns
- embedding columns
- shared embedding columns
- crossed columns

It also defines feature groups used by the model, such as:

- `wide`
- `deep`
- `cross`
- `fm`
- `attention_*`
- `pnn_*`

During training, `FeatureTransformer` reads these generated definitions and builds the actual feature-column groups consumed by the model.

### 5. CSV-based training input

The current training path uses CSV input.

The CSV reader:

- expects the first row to be the header
- maps columns by name
- requires a `label` column
- supports generated fixed-length and variable-length field definitions

This logic is implemented in `DataInput.get_dataset_from_csv_v3` in `utils/common.py`.

### 6. Training, evaluation, prediction, and export

`model_train.py` supports these run modes:

- `train`
- `eval`
- `predict`
- `export`
- `train_and_evaluate`

The training script loads:

- `FEATURE_CONFIG` from `feature_config/model_json_conf.py`
- `FEATURE_TRANSFORM_CONFIG` from `feature_config/model_input_fn.py`

Then it:

1. reads the input dataset
2. parses raw fields according to the generated schema
3. builds feature columns and feature groups
4. feeds them into the selected TensorFlow model
5. runs training or inference

## Configuration Generation Flow

The main configuration flow is:

```text
criteo_sample_feature_config_bucketized.json
    -> model_config.sh
    -> configs/config_preprocess_fields.py
    -> configs/config_train_fields.py
    -> configs/config_model_struct.py
    -> feature_config/model_json_conf.py
    -> feature_config/model_input_fn.py
    -> model_train.py
```

In practice:

- `model_json_conf.py` controls how data is parsed
- `model_input_fn.py` controls how features are transformed
- `model_train.py` uses both during training

## Verified Data

The current TensorFlow training path has been verified with:

```text
data/open_data/criteo_sample/normalized/criteo_sample.csv
```

and the corresponding feature definition:

```text
data/open_data/criteo_sample/criteo_sample_feature_config_bucketized.json
```

## Typical Workflow

### 1. Prepare or edit the raw feature JSON

Example:

```text
data/open_data/criteo_sample/criteo_sample_feature_config_bucketized.json
```

### 2. Generate feature configuration Python files

```bash
bash model_config.sh
```

### 3. Run training

```bash
python model_train.py \
  --run_mode=train \
  --train_data_dir=./data/open_data/criteo_sample/normalized \
  --eval_data_dir=./data/open_data/criteo_sample/normalized \
  --predict_data_dir=./data/open_data/criteo_sample/normalized \
  --model_dir=./model_dir \
  --export_dir=./export_dir \
  --batch_size=128 \
  --num_epochs=1
```

### 4. Optional: run prediction or export

Prediction:

```bash
python model_train.py \
  --run_mode=predict \
  --predict_data_dir=./data/open_data/criteo_sample/normalized \
  --predict_data_output_dir=./predict_output.txt \
  --model_dir=./model_dir
```

Export:

```bash
python model_train.py \
  --run_mode=export \
  --model_dir=./model_dir \
  --export_dir=./export_dir
```

## Repository Layout

```text
DeepRecKit/
|-- configs/         # scripts that generate Python config from raw JSON
|-- data/            # local sample data and raw feature config JSON
|-- feature_config/  # generated Python feature config files
|-- features/        # TensorFlow feature helper code
|-- model/           # TensorFlow model implementations
|-- utils/           # data readers and utility code
|-- model_config.sh  # generate feature_config/*.py from JSON
|-- model_train.py   # main TensorFlow training entry point
|-- model_train.sh   # shell wrapper for training
`-- pipeline/        # auxiliary scripts such as sample dataset download
```

## Auxiliary Scripts

The `pipeline/` directory is auxiliary. It can help with dataset preparation and local demos, but the main capability of this repository is still the TensorFlow training path described above.

## Current Limitations

- The main code path is TensorFlow 1.x based.
- Some model variants under `model/` are still experimental.
- The repository assumes locally prepared CSV data and generated feature config files.
