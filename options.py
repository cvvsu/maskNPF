import argparse

def get_message(parser, args):
    """https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/f13aab8148bd5f15b9eb47b690496df8dadbab0c/options/base_options.py#L88
    """
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(args).items()):
        comment = ''
        default = parser.get_default(k)
        if v != default:
            comment = '\t[default: %s]' % str(default)
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    return message


def get_args():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--ckpt_dir', default='checkpoints', type=str, help='folder stores the checkpoints')
    parser.add_argument('--dataroot', metavar='DIR', default='datasets', help='path to dataset')
    parser.add_argument('--station', default='varrio', help='station that the dataset collected from')
    parser.add_argument('--model_name', default='maskrcnn.pth', type=str, help='name of the pretrained model')
    parser.add_argument('--im_size', default=256, type=int, help='image size for training. Square sizes are used but rectangle are also possible')
    parser.add_argument('--scores', default=0.00, type=float, help='threshold for objectiveness scores')
    parser.add_argument('--vmax', default=1e4, type=float, help='value scales for drawing')
    parser.add_argument('--dynamic_vmax', action='store_true', help='utilize the dynamic surface plot if specified')
    parser.add_argument('--time_res', default=10, type=float, help='the time resolution of measurements. Default is 10 minutes')
    parser.add_argument('--mask_thres', default=0.5, type=float, help='threshold to binarize an mask')
    parser.add_argument('--ftsize', default=16, type=int, help='font size for elements in plots')
    args = parser.parse_args()

    return args, get_message(parser, args)
