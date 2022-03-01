#!/usr/bin/env python
import os
import string
import random
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)
class MyModel(nn.Module):
    """
    This is a starter model to get you started. Feel free to modify this file.
    """
    # Need to fix this because it relies on info it doesn't have technically yet
    def __init__(self, vocab_size):
        super().__init__()

        # Can keep hardcoded
        embedding_dim = 50
        # 50
        embedding_matrix = torch.normal(0, 1, (vocab_size, embedding_dim))
        # Construct embedding layer and initialize with given embedding matrix. Do not modify this code.
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=0)
        self.embedding.weight.data = embedding_matrix
        #self.embedding.weight.data.requires_grad = False
        #self.type_module = nn.RNN(50, 64, batch_first=True)
        self.type_module = nn.LSTM(embedding_dim, 64, batch_first=True)
        #self.type_module = nn.GRU(50, 64, batch_first=True)
        self.linear_module = nn.Linear(64, vocab_size)

        #raise NotImplementedError


    def forward(self, inputs):
        """
        Takes in a batch of data of shape (N, max_sequence_length). Returns a tensor of shape (N, 1), where each
        element corresponds to the prediction for the corresponding sequence.
        :param inputs: Tensor of shape (N, max_sequence_length) containing N sequences to make predictions for.
        :return: Tensor of predictions for each sequence of shape (N, 1).
        """
        first = self.embedding(inputs)
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
        targets = torch.flatten(targets)
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
        targets = torch.flatten(targets)
        targets = targets.to(device)
        output_transformed = torch.tensor([torch.argmax(output[i]) for i in range(len(output))])
        
        output_transformed = output_transformed.to(device)
        return torch.eq(output_transformed, targets).sum() / len(output_transformed)
        #raise NotImplementedError

    @classmethod
    def load_training_data(cls):
        # your code here
        # https://www.statmt.org/europarl/
        # Need to download parallel corpus Bulgarian-English and unzip before running
        batch_num = 4
        words_from_each = 100000
        with open("data/europarl-v7.bg-en.en", encoding='Latin1') as f:
            lines = f.read().lower()
            lines = lines.translate(str.maketrans('', '', string.punctuation))
            english_tokens = lines.split()
            
            english_tokens = english_tokens[:words_from_each]
            print("length of english tokens", len(english_tokens))
        with open("data/europarl-v7.bg-en.bg", encoding='utf-8') as f:
            lines = f.read().lower()
            lines = lines.translate(str.maketrans('', '', string.punctuation))
            bulgarian_tokens = lines.split()
            bulgarian_tokens = bulgarian_tokens[:words_from_each]
            print("length of bulgarian tokens", len(bulgarian_tokens))
            tokens = english_tokens + bulgarian_tokens
            print("length of input text", len(tokens))
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
            
            # print("length of vocab", len(word_to_index))
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
            # print(input_tensor.shape)
            for i in range(0, len(tokens), batch_num):
                for j in range(0, batch_num - 1):
                    input_tensor[int(i / batch_num)][j] = tokens[i + j]
                label_tensor[int(i / batch_num)][0] = tokens[i + batch_num - 1]
            data = (input_tensor, label_tensor)
        torch.save(data, "work/preprocessed_data.pt")
        index_to_word = {}
        for word in word_to_index:
            index_to_word[word_to_index[word]] = word
        torch.save(index_to_word, "work/index_to_word.pt")
        torch.save(word_to_index, "work/word_to_index.pt")
        return data, word_to_index

    @classmethod
    def load_test_data(cls, fname):
        # your code here
        data = []
        with open(fname) as f:
            for line in f:
                inp = line[:-1]  # the last character is a newline
                data.append(inp)
        for i in range(len(data)):
            tokens = data[i].lower().split()
            data[i] = tokens
        return data

    @classmethod
    def write_pred(cls, preds, fname):
        with open(fname, 'wt') as f:
            for p in preds:
                f.write('{}\n'.format(p))

    def run_train(self, data, work_dir):
        params = {'batch_size': 128,
          'shuffle': True,
          'num_workers': 2}
        training_set = self.Dataset(data[0].to(torch.int64), data[1].to(torch.int64))
        train_loader = torch.utils.data.DataLoader(training_set, **params)

        LEARNING_RATE = 0.01
        NUM_EPOCHS = 50
        optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE)
        # print(self.parameters)
        for epoch in range(NUM_EPOCHS):
                # Total loss across train data
            train_loss = 0.
                # Total number of correctly predicted training labels
            train_correct = 0
                # # Total number of training sequences processed
            train_seqs = 0

            #tqdm_train_loader = tqdm(train_loader, position=0, leave=True)
            

            self.train()
            for batch_idx, batch in enumerate(train_loader):
                sentences_batch, labels_batch = batch
                sentences_batch = sentences_batch.to(device)
                labels_batch = labels_batch.to(device)
                # print(sentences_batch.shape)

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

            avg_train_loss = train_loss / len(train_loader)
            train_accuracy = train_correct / train_seqs
            if (epoch % 10 == 0):
                print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
                print(f"[Training Loss]: {avg_train_loss:.4f} [Training Accuracy]: {train_accuracy:.4f}")
            #pass

    def run_pred(self, data):
        # your code here
        all_chars = string.ascii_letters
        preds = []
        last_words = []
        for i in range(len(data)):
            last_words.append(data[i][-1])
            data[i].pop()
        index_to_word = torch.load("work/index_to_word.pt")
        word_to_index = torch.load("work/word_to_index.pt")
        self.eval()
        for i in range(len(data)):
            words_that_matter_indices = []
            last_word = last_words[i]
            for token in word_to_index:
                if token.startswith(last_word) and len(token) > len(last_word):
                    words_that_matter_indices.append(word_to_index[token])
            single_test_item = data[i]
            for j in range(len(single_test_item)):
                if single_test_item[j] in word_to_index:
                    single_test_item[j] = word_to_index[single_test_item[j]]
                else:
                    single_test_item[j] = word_to_index['UNK']
            if len(single_test_item) == 0:
                single_test_item.append(word_to_index['UNK'])
            single_tensor = torch.tensor(single_test_item)
            single_tensor = single_tensor.long()
            single_tensor = torch.reshape(single_tensor, (1, len(single_tensor)))
            with torch.no_grad():
                output = self(single_tensor)
                output = torch.flatten(output)
                output = output[words_that_matter_indices]
                char_to_output_dict = {}
                for i in range(len(words_that_matter_indices)):
                    char_to_output_dict[words_that_matter_indices[i]] = output[i]
                sorted_dict = dict(sorted(char_to_output_dict.items(), key=lambda item: item[1], reverse=True))
                prediction_set = set()
                for word_index in sorted_dict:
                    word = index_to_word[word_index]
                    pred_char = word[len(last_word)]
                    prediction_set.add(pred_char)
                    if (len(prediction_set) == 3):
                        break
                while len(prediction_set) < 3:
                    prediction_set.add(random.choice(all_chars).lower())
                preds.append(''.join(list(prediction_set)))
        return preds

    def save(self, work_dir):
        # your code here
        # this particular model has nothing to save, but for demonstration purposes we will save a blank file
        # with open(os.path.join(work_dir, 'model.checkpoint'), 'wt') as f:
        #     f.write('dummy save')
        torch.save(self.state_dict(), work_dir + "/model.checkpoint")

    @classmethod
    def load(cls, work_dir):
        # your code here
        # this particular model has nothing to load, but for demonstration purposes we will load a blank file
        # with open(os.path.join(work_dir, 'model.checkpoint')) as f:
        #     dummy_save = f.read()
        index_to_word = torch.load("work/index_to_word.pt")
        model = MyModel(len(index_to_word))
        model.load_state_dict(torch.load(work_dir + "/model.checkpoint"))
        return model

    class Dataset(torch.utils.data.Dataset):
        def __init__(self, features, labels):
            self.labels = labels
            self.features = features

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, index):
            return self.features[index], self.labels[index]


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
        # print('Loading training data')
        # MyModel.load_training_data()
        train_data = torch.load("work/preprocessed_data.pt")
        index_to_word = torch.load("work/index_to_word.pt")
        print(len(index_to_word))
        print('Instatiating model')
        model = MyModel(len(index_to_word))
        model.to(device)
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
