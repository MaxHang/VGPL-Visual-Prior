"""Package configurations."""

import os

from utils import CfgNode


def get_default():
    """Get default configurations."""

    config = CfgNode()

    # Experiment
    # config.run_dir = './experiments/test_run'
    # config.run_dir = '/workspace/VGPL-Visual-Prior/experiments/cconv_fluid_801_v1'
    config.run_dir = '/workspace/VGPL-Visual-Prior/experiments/cconv_fluid_v4'
    config.load_path = ''
    config.debug = False

    # config.dataset = 'RigidFall'  # 'RigidFall', 'MassRope' 'CConvFluid'
    config.dataset = 'CConvFluid'  # 'RigidFall', 'MassRope' 'CConvFluid'
    # config.dataset = 'CConvFluid801times'  # 'RigidFall', 'MassRope' 'CConvFluid'
    # config.dataset = 'MassRope'  # 'RigidFall', 'MassRope'
    # config.batch_size = 128
    config.batch_size = 50

    # config.n_frames = 1
    config.n_frames = 4
    config.loss_type = 'l2'  # 'l2', 'chamfer'
    # config.loss_type = 'chamfer'  # 'l2', 'chamfer'
    config.group_loss_weight = 1.0  # loss = pos_loss + w * grp_loss
    # config.group_loss_weight = 0.0  # loss = pos_loss + w * grp_loss
    config.lr = 1e-4
    config.n_epochs = 50

    config.batchnorm = True  # batch norm on new model

    config.recur_pred = False  # use recurrent predictor
    config.use_temp_encoder = False  # use temporal encoder
    config.conv_temp_encoder = False  # use convolutional temporal encoder
    config.temp_embedding_size = 1024
    config.single_out = False
    config.pred_hidden = 2048

    config.temp_reg_lam = 0.0  # regularization factor of temporal smoothness

    config.vis = True
    config.vis_every = 5
    config.vis_samples = 5
    # config.vis_frames_per_sample = 64
    config.vis_frames_per_sample = 15

    config.n_frames_eval = 10
    # config.n_frames_eval = 30
    config.log_eval = True
    config.vis_eval = True
    config.vis_eval_every = 50

    config.cal_vel = False   # cal point velocity
    config.cal_vel_time_dif = 1 / 50.0   # cal point velocity

    return config


def set_params(config, file_path=None, list_opt=None):
    """
    Set config parameters with config file and options.
    Option list (usually from command line) has the highest
    overwrite priority.
    """
    if file_path:
        print('- Import config from file {}.'.format(file_path))
        config.merge_from_file(file_path)
    if list_opt:
        print('- Overwrite config params {}.'.format(str(list_opt[::2])))
        config.merge_from_list(list_opt)
    return config


def freeze(config, save_file=False):
    """Freeze configuration and save to file (optional)."""
    config.freeze()
    if save_file:
        if not os.path.isdir(config.run_dir):
            os.makedirs(config.run_dir)
        with open(os.path.join(config.run_dir, 'config.yaml'), 'w') as fout:
            fout.write(config.dump())
