#!/usr/bin/env python
import os
import string
import random
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import torch

class MyModel:
    """
    This is a starter model to get you started. Feel free to modify this file.
    """

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
            print(tokens[:100])
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
            # We then need to make the inputs one hot encoding
            # tokens should be a tuple of ((9845304 / 4, 3, 26603), (9845304 / 4, 1))
            input_tensor = torch.zeros(( int(len(tokens) / batch_num), batch_num - 1))
            label_tensor = torch.zeros((int(len(tokens) / batch_num), 1))
            print(input_tensor.shape)
            for i in range(0, len(tokens), batch_num):
                for j in range(0, batch_num - 1):
                    input_tensor[int(i / batch_num)][j] = tokens[i + j]
                label_tensor[int(i / batch_num)][0] = tokens[i + batch_num - 1]
            data = (input_tensor, label_tensor)
            #print(data[0][:5])
        return []

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
    parser.add_argument('mode', choices=('train', 'test'), help='what to run')
    parser.add_argument('--work_dir', help='where to save', default='work')
    parser.add_argument('--test_data', help='path to test data', default='example/input.txt')
    parser.add_argument('--test_output', help='path to write test predictions', default='pred.txt')
    args = parser.parse_args()

    random.seed(0)

    if args.mode == 'train':
        if not os.path.isdir(args.work_dir):
            print('Making working directory {}'.format(args.work_dir))
            os.makedirs(args.work_dir)
        print('Instatiating model')
        model = MyModel()
        print('Loading training data')
        train_data = MyModel.load_training_data()
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
