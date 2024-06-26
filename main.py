import os
import numpy as np
import tensorflow as tf
import torch

import keras
from torch import nn

from utility.file_utility import FileUtility

acid_letters = "abcdefghiklmnpqrstuvwxyz"


class MLP_OF(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        # Create model with specified layers
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(len(acid_letters) + 1, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.CELU(),
            nn.Linear(16, 1),
            # DROPOUT LAYER?
            # Different activation
            nn.Sigmoid()
        )
        # one hot encoding (lib functions)

    def forward(self, x):
        x = torch.FloatTensor(x)
        x = self.linear_relu_stack(x)
        return x


def encode_seq(seq):
    amino_counts = {
        char: 0 for char in acid_letters
    }

    for character in seq.lower():
        amino_counts[character] += 1

    ret_list = list(amino_counts.values())
    ret_list.append(len(seq))

    return ret_list


def one_hot_encode(seq: str):
    seq = seq.lower()
    encoded = []
    max_characters = 128

    for character in seq:
        one_hot_vector = np.zeros(len(acid_letters))
        one_hot_vector[acid_letters.index(character)] = 1
        encoded.append(one_hot_vector)
        if len(encoded) == max_characters:
            break

    while len(encoded) < max_characters:
        encoded.append(np.zeros(len(acid_letters)))

    return encoded


def encode_seq_list(sequence_list):
    encoded_list = []
    for seq in sequence_list:
        encoded_list.append(encode_seq(seq))

    return encoded_list


def train_model(model: MLP_OF, training_set, validation_set):
    loss = nn.BCELoss()
    vloss = nn.BCELoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.00001, momentum=0.9)

    best_loss = 1_000_000

    epoch = 0

    while best_loss > 0.2:
        model.train(True)
        # TRAIN ONE EPOCH
        running_loss = 0.0

        for seq, label in training_set:
            optimizer.zero_grad()
            output = model(seq)
            loss_out = loss(output, torch.from_numpy(np.array(label, dtype=np.float32).reshape(1, )))
            loss_out.backward()
            optimizer.step()
            running_loss += loss_out.item()

        # EVALUATE TRAINING
        # TODO: EVAL AFTER TRAINING
        model.eval()
        running_vloss = 0.0

        with torch.no_grad():
            for vseq, vlabel in validation_set:
                output = model(vseq)
                vloss_out = vloss(output, torch.from_numpy(np.array(vlabel, dtype=np.float32).reshape(1, )))
                running_vloss += vloss_out.item()

        print("E: {}; RunLoss: {}; RunVLoss: {}".format(epoch, running_loss, running_vloss))

        if running_vloss < best_loss:
            best_loss = running_vloss
            model_path = 'models/model_{}_best.pt'.format(epoch)
            if len(os.listdir("models")) > 0:
                for entry in os.listdir("models"):
                    if entry.endswith(".pt"):
                        os.remove(os.path.join("models", entry))
            torch.save(model.state_dict(), model_path)

        epoch += 1


def validate(model, validation_set):
    model.eval()

    for seq, label in validation_set:
        output = model(seq)
        print("Expected: {}; Actual: {}".format(label, output.item()))


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

    training_set = list(zip(encoded_train, y_train))

    encoded_test = encode_seq_list(seq_test)

    validation_set = list(zip(encoded_test, y_test))

    model = MLP_OF()

    model.to(device)

    train_model(model, training_set, validation_set)

    validate(model, validation_set)


if __name__ == '__main__':
    main()
