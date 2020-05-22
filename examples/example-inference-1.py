#!/usr/bin/env python3

import argparse
import os
import sys
import time

import tensorflow as tf
import tensorlayer as tl

sys.path.append('../')

from openpose_plus.inference.common import measure, plot_humans, read_imgfile
from openpose_plus.inference.estimator import TfPoseEstimator
from openpose_plus.models import get_model
# from get_files_path_list.py import get_file_path_list

tf.logging.set_verbosity(tf.logging.INFO)
tl.logging.set_verbosity(tl.logging.INFO)


def get_file_path_list(path='.'):
    images = []
    for dirname, dirnames, filenames in os.walk(path):
        # print path to all subdirectories first.
        for subdirname in dirnames:
            print(os.path.join(dirname, subdirname))

        # print path to all filenames.
        for filename in filenames:
            # print(os.path.join(dirname, filename))
            images += [os.path.abspath(os.path.join(dirname, filename))]

        # Advanced usage:
        # editing the 'dirnames' list will stop os.walk() from recursing into there.
        if '.git' in dirnames:
            # don't go into any .git directories.
            dirnames.remove('.git')
    return ','.join(images)


def inference(base_model_name, path_to_npz, data_format, input_files, plot):
    model_func = get_model(base_model_name)
    height, width = (368, 432)
    e = measure(lambda: TfPoseEstimator(path_to_npz, model_func, target_size=(width, height), data_format=data_format),
                'create TfPoseEstimator')

    t0 = time.time()
    for idx, img_name in enumerate(input_files):
        image = measure(lambda: read_imgfile(img_name, width, height, data_format=data_format), 'read_imgfile')
        humans, heatMap, pafMap = measure(lambda: e.inference(image), 'e.inference')
        tl.logging.info('got %d humans from %s' % (len(humans), img_name))
        if humans:
            for h in humans:
                tl.logging.debug(h)
        if plot:
            if data_format == 'channels_first':
                image = image.transpose([1, 2, 0])
            plot_humans(image, heatMap, pafMap, humans, '%02d' % (idx + 1),'inferece')
    tot = time.time() - t0
    mean = tot / len(input_files)
    tl.logging.info('inference all took: %f, mean: %f, FPS: %f' % (tot, mean, 1.0 / mean))


def parse_args():
    parser = argparse.ArgumentParser(description='inference')
    parser.add_argument('--path-to-npz', type=str, default='', help='path to npz', required=True)
    parser.add_argument('--images', type=str, default='../data/your_data/images', help='comma separate list of image filenames', required=False)
    parser.add_argument('--base-model', type=str, default='hao28_experimental', help='hao28_experimental | vgg | vggtiny | mobilenet')
    parser.add_argument('--data-format', type=str, default='channels_last', help='channels_last | channels_first.')
    parser.add_argument('--plot', type=bool, default=False, help='draw the results')
    parser.add_argument('--repeat', type=int, default=1, help='repeat the images for n times for profiling.')
    parser.add_argument('--limit', type=int, default=1000, help='max number of images.')

    return parser.parse_args()


def main():
    args = parse_args()
    image_files = ([f for f in get_file_path_list(args.images).split(',') if f] * args.repeat)[:args.limit]
    inference(args.base_model, args.path_to_npz, args.data_format, image_files, args.plot)


if __name__ == '__main__':
    measure(main)
