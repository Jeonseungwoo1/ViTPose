from ViTPose_common import *

channel_cfg = dict(
    num_output_channels=16,
    dataset_joints=16,
    dataset_channel=list(range(16)),
    inference_channel=list(range(16)))

data_cfg['num_output_channels'] =channel_cfg['num_output_channels']
data_cfg['num_joints'] =channel_cfg['dataset_joints']
data_cfg['dataset_channel'] =channel_cfg['dataset_channel']
data_cfg['inference_channel'] =channel_cfg['inference_channel']


names = ['small', 'base', 'large', 'huge']
for name in names:
    globals()[f'model_{name}']['keypoint_head']['out_channels'] = channel_cfg['num_output_channels'] 