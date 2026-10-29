"""Arguments required by the original self-supervised training scripts."""

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--selected_dataset', default='EEG')
parser.add_argument('--selected_method', default='CPC')
parser.add_argument('--training_mode', default='self_supervised',
                    choices=('self_supervised', 'fine_tune'))
parser.add_argument('--experiment_description', default='DINOv3_EEG')
parser.add_argument('--logs_save_dir', default='experiments_logs')
parser.add_argument('--device', default='cuda')
parser.add_argument('--use_SimCLR', action='store_true')
args = parser.parse_args()
