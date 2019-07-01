#!/usr/bin/env python3
"""Export pre-trained openpose model for C++/TensorRT."""

import argparse
import os
import sys

import tensorflow as tf
import tensorlayer as tl

sys.path.append('.')

from openpose_plus.inference.common import measure, rename_tensor
from openpose_plus.models import get_model

from train_config import config

tf.logging.set_verbosity(tf.logging.DEBUG)
tl.logging.set_verbosity(tl.logging.DEBUG)


def mkdir_p(full_path):
    os.makedirs(full_path, exist_ok=True)


def save_graph(sess, checkpoint_dir, name):
    tf.train.write_graph(sess.graph_def, checkpoint_dir, name)


def save_model(sess, checkpoint_dir, global_step=0):
    saver = tf.train.Saver()
    checkpoint_prefix = os.path.join(checkpoint_dir, "saved_checkpoint")
    checkpoint_state_name = 'checkpoint_state'
    saver.save(sess, checkpoint_prefix, global_step=global_step, latest_filename=checkpoint_state_name)


def save_uff(sess, names, filename):
    import uff
    frozen_graph = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, names)
    tf_model = tf.graph_util.remove_training_nodes(frozen_graph)
    uff.from_tensorflow(tf_model, names, output_filename=filename)


def export_model(model_func, checkpoint_dir, path_to_npz, graph_filename, uff_filename):
    model_parameters = model_func()
    names = [p.name[:-2] for p in model_parameters]
    print('name: %s' % ','.join(names))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        measure(lambda: tl.files.load_and_assign_npz_dict(path_to_npz, sess), 'load npz')

        if graph_filename:
            mkdir_p(checkpoint_dir)
            measure(lambda: save_graph(sess, checkpoint_dir, graph_filename), 'save_graph')
            measure(lambda: save_model(sess, checkpoint_dir), 'save_model')

        if uff_filename:
            measure(lambda: save_uff(sess, names, uff_filename), 'save_uff')

    print('exported model_parameters:')
    for p in model_parameters:
        print('%s :: %s' % (p.name, p.shape))

def main():

    def model_func():
        target_size = (config.MODEL.win, config.MODEL.hin)
        return get_model(config.MODEL.name)(target_size, config.MODEL.data_format)

    export_model(model_func, config.EXPORT.checkpoint_dir, os.path.join(config.MODEL.model_path, config.EXPORT.model), config.EXPORT.graph_filename, config.EXPORT.uff_filename)


if __name__ == '__main__':
    main()
