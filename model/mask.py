#!/usr/bin/python
# -*- encoding: utf-8 -*-
# @Author: gavinlzhang
# @Date: 2022/9/22
import tensorflow as tf

def mask_by_conditions(tensor, conditions=None):
    if conditions is None:
        conditions = []
    res = tensor
    for cond in conditions:
        res = tf.where(cond, res, tf.zeros_like(res))
    return res
