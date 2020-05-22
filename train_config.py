import os

from easydict import EasyDict as edict

config = edict()

config.TRAIN = edict()
config.TRAIN.batch_size = 10
config.TRAIN.save_interval = 2500
config.TRAIN.log_interval = 10
#config.TRAIN.n_epoch = #10
config.TRAIN.n_step = 600000  # total number of step
config.TRAIN.lr_init = 1.641157e-07  #4e-5 # initial learning rate
config.TRAIN.lr_decay_every_step = 100000  # evey number of step to decay lr
config.TRAIN.lr_decay_factor = 0.333  # decay lr factor
config.TRAIN.weight_decay_factor = 5e-4#5e-4
config.TRAIN.train_mode = 'single'  # single, parallel
config.TRAIN.train_batch_norm = False  # should bn layers be trained or kept
config.TRAIN.shuffel_buffer_size = 10000

config.MODEL = edict()
config.MODEL.model_path = 'models'  # save directory
config.MODEL.model_file = 'pose.npz'  # save file
config.MODEL.n_pos = 19  # number of keypoints + 1 for background
config.MODEL.hin = 368#256  # input size during training , 240
config.MODEL.win = 368#384
config.MODEL.hout = int(config.MODEL.hin / 8)  # output size during training (default 46)
config.MODEL.wout = int(config.MODEL.win / 8)
config.MODEL.name = 'hao28_experimental'  # vgg, vggtiny, mobilenet, hao28_experimental

config.MODEL.initial_weights = True  # True,False
config.MODEL.initial_weights_file = 'pose_best_67726_2.118974208831787_.npz' # 'hao28-pose600000.npz' # 'mobilenet.npz'  # save file

config.MODEL.data_format='channels_first' #'channels_last' or 'channels_first'


if (config.MODEL.hin % 16 != 0) or (config.MODEL.win % 16 != 0):
    raise Exception("image size should be divided by 16")

config.DATA = edict()
config.DATA.train_data = 'custom'  # coco, custom, coco_and_custom
config.DATA.coco_version = '2017'  # MSCOCO version 2014 or 2017
config.DATA.data_path = 'data'
config.DATA.your_images_path = os.path.join('data', 'your_data', 'images')
config.DATA.your_annos_path = os.path.join('data', 'your_data', 'coco.json')

config.LOG = edict()
config.LOG.vis_path = 'vis'

config.EVAL = edict()
config.EVAL.model = 'pose_best.npz'
config.EVAL.eval_path = 'eval'
config.EVAL.data_idx = -1 # data_idx >= 0 to use specified data
config.EVAL.eval_size = -1 # use first eval_size elements to evaluate, only when data_idx < 0
config.EVAL.plot = True

config.EXPORT = edict()
config.EXPORT.model = 'pose_export.npz'
config.EXPORT.graph_filename = None
config.EXPORT.checkpoint_dir = None
config.EXPORT.uff_filename = "{}_w{}xh{}.uff".format(config.MODEL.name,config.MODEL.win,config.MODEL.hin)
