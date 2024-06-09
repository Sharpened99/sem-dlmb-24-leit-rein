import torch

import keras
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


def train_model(model, training_set, validation_set):
    loss_fn = keras.losses.binary_crossentropy

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    best_loss = 1_000_000

    for epoch in range(100):
        model.train(True)
        # TRAIN ONE EPOCH
        running_loss = 0.0
        optimizer.zero_grad()

        for seq, label in training_set:
            output = model(seq)
            loss = loss_fn(output, label)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # EVALUATE TRAINING
        model.eval()
        running_vloss = 0.0

        with torch.no_grad():
            for vseq, vlabel in validation_set:
                output = model(vseq)
                vloss = loss_fn(output, vlabel)
                running_vloss += vloss.item()

        if running_vloss < best_loss:
            best_loss = running_vloss
            model_path = 'model_{}_best.pt'.format(epoch)
            torch.save(model.state_dict(), model_path)


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

    y_train = [float(number) for number in y_train]

    y_test = [float(number) for number in y_test]

    encoded_train = encode_seq_list(seq_train)

    training_set = zip(encoded_train, y_train)

    encoded_test = encode_seq_list(seq_test)

    validation_set = zip(encoded_test, y_test)

    model = MLP()

    model.to(device)

    train_model(model, training_set, validation_set)


if __name__ == '__main__':
    main()
