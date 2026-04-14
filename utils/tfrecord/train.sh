#!/usr/bin/env bash
# set environment variables (if not already done)
export PYTHON_ROOT=./Python
export LD_LIBRARY_PATH=${PATH}
export PYTHONPATH=Python/bin/python3
#export PYSPARK_PYTHON=python3
export PYSPARK_PYTHON=Python/bin/python3
export PYSPARK_DRIVER_PYTHON=Python/bin/python3
export SPARK_YARN_USER_ENV="PYSPARK_PYTHON=Python/bin/python3"
export PATH=${PYTHON_ROOT}/bin/:$PATH
export QUEUE=airec

# set paths to libjvm.so, libhdfs.so, and libcuda*.so
#export LIB_HDFS=/opt/cloudera/parcels/CDH/lib64                      # for CDH (per @wangyum)
export LIB_HDFS=$HADOOP_HOME/lib/native              # path to libhdfs.so, for TF acccess to HDFS
export LIB_JVM=$JAVA_HOME/jre/lib/amd64/server                        # path to libjvm.so
#export LIB_CUDA=/usr/local/cuda-7.5/lib64                             # for GPUs only

# for CPU mode:
# export QUEUE=default
# remove references to $LIB_CUDA

# save images and labels as CSV files
${SPARK_HOME}/bin/spark-submit \
--master yarn \
--deploy-mode cluster \
--queue ${QUEUE} \
--num-executors 1 \
--executor-memory 2G \
--conf spark.dynamicAllocation.enabled=false \
--conf spark.yarn.maxAppAttempts=1 \
--py-files /home/zhanglanhui/tfos/tfspark.zip \
--archives hdfs:///recommend/python/Python.zip#Python \
/home/zhanglanhui/tfos/gen_tfrecords.py
