#!/usr/bin/env python
import os
import string
import random
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import torch
import torch.nn as nn
import torch.optim as optim


class MyModel(nn.Module):
    """
    This is a starter model to get you started. Feel free to modify this file.
    """
    def __init__(self, training_data):
        super().__init__()

        # Change and pass in the vocab
        vocab_size = 6629
        # 6629
        # Can keep hardcoded
        embedding_dim = 50
        # 50
        embedding_matrix = torch.normal(0, 1, (vocab_size, embedding_dim))
        print(embedding_matrix.dtype)
        # Construct embedding layer and initialize with given embedding matrix. Do not modify this code.
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=0)
        self.embedding.weight.data = embedding_matrix
        print(self.embedding.weight)
        #self.embedding.weight.data.requires_grad = False
        #self.type_module = nn.RNN(50, 64, batch_first=True)
        self.type_module = nn.LSTM(embedding_dim, vocab_size, batch_first=True)
        #self.type_module = nn.GRU(50, 64, batch_first=True)
        self.linear_module = nn.Linear(vocab_size, 1)

        #raise NotImplementedError


    def forward(self, inputs):
        """
        Takes in a batch of data of shape (N, max_sequence_length). Returns a tensor of shape (N, 1), where each
        element corresponds to the prediction for the corresponding sequence.
        :param inputs: Tensor of shape (N, max_sequence_length) containing N sequences to make predictions for.
        :return: Tensor of predictions for each sequence of shape (N, 1).
        """
        print("starting forward")
        first = self.embedding(inputs)
        print("Finished embedding")
        output, hidden = self.type_module(first)
        output = output[:, -1:, :]
        output = torch.reshape(output, (output.shape[0], output.shape[2]))
        third = self.linear_module(output)
        return third

    def loss(self, output, targets):
        """
        Computes the binary cross-entropy loss.
        :param logits: Raw predictions from the model of shape (N, 1)
        :param targets: True labels of shape (N, 1)
        :return: Binary cross entropy loss between logits and targets as a scalar tensor.
        """
        loss_function = nn.CrossEntropyLoss()
        result = loss_function(output, targets)
        return result
        #raise NotImplementedError

    def accuracy(self, output, targets):
        """
        Computes the accuracy, i.e number of correct predictions / N.
        :param logits: Raw predictions from the model of shape (N, 1)
        :param targets: True labels of shape (N, 1)
        :return: Accuracy as a scalar tensor.
        """
        
        output_transformed = torch.tensor([torch.argmax(output[i]) for i in range(len(output))])
        return torch.eq(output_transformed, targets).sum() / len(output)
        #raise NotImplementedError

    @classmethod
    def load_training_data(cls):
        # your code here
        # this particular model doesn't train
        # https://www.statmt.org/europarl/
        # Need to download parallel corpus Bulgarian-English and unzip before running
        with open("europarl-v7.bg-en.en", encoding='Latin1') as f:
            # Supposed to split into sentences but not doing that for now, also going to ignore padding and just take off the last couple words
            lines = f.read().lower()
            lines = lines.translate(str.maketrans('', '', string.punctuation))
            tokens = lines.split()
            tokens = tokens[:int(len(tokens) / 26)]
            token_dict = dict()
            for token in tokens:
                if token in token_dict:
                    token_dict[token] += 1
                else:
                    token_dict[token] = 1
            unknown_list = set()
            for token in token_dict:
                if token_dict[token] < 3:
                    unknown_list.add(token)
            for item in unknown_list:
                token_dict.pop(item)
            token_dict['UNK'] = 1
            for i in range(len(tokens)):
                if tokens[i] not in token_dict:
                    tokens[i] = 'UNK'
            word_to_index = {}
            index = 0
            for token in token_dict:
                word_to_index[token] = index
                index += 1
            for i in range(len(tokens)):
                tokens[i] = word_to_index[tokens[i]]
            #print(tokens[:300])
            tokens = tokens[:-2]
            batch_num = 4
            print("length of vocab", len(word_to_index))
            print("length of input text", len(tokens))
            # We have the words tokenized and replaced uncommon words with unk
            # We don't have to worry about padding for this example since its already divisible by 4
            # https://towardsdatascience.com/exploring-the-next-word-predictor-5e22aeb85d8f
            # Using that link on how to do the preprocessing.
            # Next Steps:
            # Convert tokens into a 2d matrix where we split it up into sequences of words of batch size
            # words in text = 9845304, batch size = 4, vocab = 26603
            # tokens should become size (9845304 / 4, 4)
            # Then we take each group of 4 and remove the 4th word and make the label
            # tokens should be a tuple of ((9845304 / 4, 3), (9845304 / 4, 1)) representing input and labels
            input_tensor = torch.zeros(( int(len(tokens) / batch_num), batch_num - 1))
            label_tensor = torch.zeros((int(len(tokens) / batch_num), 1))
            print(input_tensor.shape)
            for i in range(0, len(tokens), batch_num):
                for j in range(0, batch_num - 1):
                    input_tensor[int(i / batch_num)][j] = tokens[i + j]
                label_tensor[int(i / batch_num)][0] = tokens[i + batch_num - 1]
            data = (input_tensor, label_tensor)
        torch.save(data, "preprocessed_data.pt")
        return data, word_to_index

    @classmethod
    def load_test_data(cls, fname):
        # your code here
        data = []
        with open(fname) as f:
            for line in f:
                inp = line[:-1]  # the last character is a newline
                data.append(inp)
        return data

    @classmethod
    def write_pred(cls, preds, fname):
        with open(fname, 'wt') as f:
            for p in preds:
                f.write('{}\n'.format(p))

    def run_train(self, data, work_dir):
        # your code here
        # print(data[0][:5])
        # print(data[1][:5])
        LEARNING_RATE = 0.01
        NUM_EPOCHS = 2
        optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE)
        print(self.parameters)
        for epoch in range(NUM_EPOCHS):
                # Total loss across train data
            train_loss = 0.
                # Total number of correctly predicted training labels
                # train_correct = 0
                # # Total number of training sequences processed
                # train_seqs = 0

            #tqdm_train_loader = tqdm(train_loader, position=0, leave=True)
            print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")

            self.train()
            # for batch_idx, batch in enumerate(tqdm_train_loader):
            #     sentences_batch, labels_batch = batch
            #     sentences_batch = sentences_batch.to(device)
            #     labels_batch = labels_batch.to(device)

            # Make predictions

            sentences_batch = data[0].to(torch.int64)
            labels_batch = data[1].to(torch.int64)

            output = self(sentences_batch)
            
            # Compute loss and number of correct predictions
            loss = self.loss(output, labels_batch)
            
            correct = self.accuracy(output, labels_batch).item() * len(output)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate metrics and update status
            train_loss += loss.item()
            train_correct += correct
            train_seqs += len(sentences_batch)
        # tqdm_train_loader.set_description_str(
        #     f"[Loss]: {train_loss / (batch_idx + 1):.4f} [Acc]: {train_correct / train_seqs:.4f}")
        # print()

        # avg_train_loss = train_loss / len(tqdm_train_loader)
        avg_train_loss = train_loss / 1
        train_accuracy = train_correct / train_seqs
        print(f"[Training Loss]: {avg_train_loss:.4f} [Training Accuracy]: {train_accuracy:.4f}")
        pass

    def run_pred(self, data):
        # your code here
        preds = []
        all_chars = string.ascii_letters
        for inp in data:
            # this model just predicts a random character each time
            top_guesses = [random.choice(all_chars) for _ in range(3)]
            preds.append(''.join(top_guesses))
        return preds

    def save(self, work_dir):
        # your code here
        # this particular model has nothing to save, but for demonstration purposes we will save a blank file
        with open(os.path.join(work_dir, 'model.checkpoint'), 'wt') as f:
            f.write('dummy save')

    @classmethod
    def load(cls, work_dir):
        # your code here
        # this particular model has nothing to load, but for demonstration purposes we will load a blank file
        with open(os.path.join(work_dir, 'model.checkpoint')) as f:
            dummy_save = f.read()
        return MyModel()


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', choices=('train', 'test', 'load_training_data'), help='what to run')
    parser.add_argument('--work_dir', help='where to save', default='work')
    parser.add_argument('--test_data', help='path to test data', default='example/input.txt')
    parser.add_argument('--test_output', help='path to write test predictions', default='pred.txt')
    args = parser.parse_args()

    random.seed(0)
    if args.mode == 'load_training_data':
        print('Loading training data')
        train_data, word_to_index = MyModel.load_training_data()
    elif args.mode == 'train':
        if not os.path.isdir(args.work_dir):
            print('Making working directory {}'.format(args.work_dir))
            os.makedirs(args.work_dir)
        train_data = torch.load("preprocessed_data.pt")
        print('Instatiating model')
        model = MyModel(train_data)  
        print('Training')
        model.run_train(train_data, args.work_dir)
        print('Saving model')
        model.save(args.work_dir)
    elif args.mode == 'test':
        print('Loading model')
        model = MyModel.load(args.work_dir)
        print('Loading test data from {}'.format(args.test_data))
        test_data = MyModel.load_test_data(args.test_data)
        print('Making predictions')
        pred = model.run_pred(test_data)
        print('Writing predictions to {}'.format(args.test_output))
        assert len(pred) == len(test_data), 'Expected {} predictions but got {}'.format(len(test_data), len(pred))
        model.write_pred(pred, args.test_output)
    else:
        raise NotImplementedError('Unknown mode {}'.format(args.mode))
