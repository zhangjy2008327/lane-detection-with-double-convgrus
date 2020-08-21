import argparse
import os
img_width = 256
img_height = 128
img_channel = 3
label_width = 256
label_height = 128
label_channel = 1
data_loader_numworkers = 8
class_num = 2
val_path = 'val_data_0531.json'
train_path = 'all_label.json'
test_path = 'test_label.json'

pretrained_path = ''
save_path = './Radam/' + pretrained_path[:12] + '/'
os.makedirs(save_path, exist_ok=True)
json_path = pretrained_path[:10] + '.json'
choose_path = './Radam/choose/'
os.makedirs(choose_path, exist_ok=True)

class_weight = [0.146, 0.99]#
# class_weight = [0.2, 1.02]#
def args_setting():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch UNet-ConvLSTM')
    parser.add_argument('--model',type=str, default='UNet_TwoConvGRU',help='( UNet-ConvLSTM | SegNet-ConvLSTM | UNet | SegNet | ')
    parser.add_argument('--batch-size', type=int, default=15, metavar='N',
                        help='input batch size for training (default: 10)')
    parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',
                        help='input batch size for testing (default: 100)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 30)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='use CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    return args
