import pandas as pd
import torch

from torch import nn

from utility.file_utility import FileUtility


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        # Create model with specified layers,
        self.linear_relu_stack = nn.Sequential(
            nn.Linear()
        )


def main():
    # train data
    seq_train = FileUtility.load_list('dataset/train_seq.txt')
    y_train = FileUtility.load_list('dataset/train_label.txt')
    # test data
    seq_test = FileUtility.load_list('dataset/test_seq.txt')
    y_test = FileUtility.load_list('dataset/test_label.txt')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    len_train = [len(s) for s in seq_train]

    print("max len: {}".format(max(len_train)))

    model = None


if __name__ == '__main__':
    main()
