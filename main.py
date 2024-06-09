import torch

from torch import nn

from utility.file_utility import FileUtility


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        # Create model with specified layers
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(27, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = torch.FloatTensor(x)
        x = self.linear_relu_stack(x)
        return x


def encode_seq(seq):
    amino_counts = {
        char: 0 for char in "abcdefghijklmnopqrstuvwxyz"
    }

    for character in seq.lower():
        amino_counts[character] += 1

    ret_list = list(amino_counts.values())
    ret_list.append(len(seq))

    return ret_list


def encode_seq_list(sequence_list):
    encoded_list = []
    for seq in sequence_list:
        encoded_list.append(encode_seq(seq))

    return encoded_list


def main():
    # train data
    seq_train = FileUtility.load_list('dataset/train_seq.txt')
    y_train = FileUtility.load_list('dataset/train_label.txt')
    # test data
    seq_test = FileUtility.load_list('dataset/test_seq.txt')
    y_test = FileUtility.load_list('dataset/test_label.txt')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # len_train = [len(s) for s in seq_train]
    # print("max len: {}".format(max(len_train)))

    encoded_train = encode_seq_list(seq_train)
    encoded_test = encode_seq_list(seq_test)

    model = MLP()

    model.to(device)

    model.train()


if __name__ == '__main__':
    main()
