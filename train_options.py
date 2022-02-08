import argparse

from parso import parse

class TrainOptions():
    def __init__(self) :
        parser = argparse.ArgumentParser(description = 'Options for training and data processing')
        parser.add_argument('--mask_root', required = True)
        parser.add_argument('--x_ray_root', required = True)
        parser.add_argument('--image_size', type = tuple, default = (128, 128))
        parser.add_argument('--in_channels', type = int, default = 1)

        parser.add_argument('--num_workers', type = int, default = 2)
        parser.add_argument('--device', type = str, default = 'cuda', help = '[cuda | cpu]')
        
        parser.add_argument('--num_epochs', type = int, default = 30)
        parser.add_argument('--num_epochs_decay', type = int, default = 0)
        parser.add_argument('--batch_size', type = int, default = 1)
        parser.add_argument('--learning_rate', type = float, default = 2e-4)
        parser.add_argument('--beta1', type = float, default = 0.5)
        parser.add_argument('--g_accumulation_steps', type = int, default = 1)
        parser.add_argument('--d_accumulation_steps', type = int, default = 1)
        
        parser.add_argument('--num_residual_blocks', type = int, default = 6)
        parser.add_argument('--lambda_cycle', type = float, default = 10.0)
        parser.add_argument('--lambda_identity', type = float, default = 0.5)

        parser.add_argument('--validate_image_path', default = None)
        parser.add_argument('--validate_frequency', type = int, default = 1)
        
        self.parser = parser

    def gather_options(self):
        return self.parser.parse_args()
    

