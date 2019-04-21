import codecs
import os

import keras
import numpy
from keras_wc_embd import get_batch_input, get_dicts_generator, get_embedding_weights_from_file
from paper2.model import build_model

from ner import NER


class UOI(NER):
    MODEL_PATH = 'model.h5'
    WORD_EMBD_PATH = 'dataset/glove.6B.100d.txt'
    RNN_NUM = 16
    RNN_UNITS = 32

    BATCH_SIZE = 16
    EPOCHS = 10
    TAGS = {
        'O': 0,
        'B-PER': 1,
        'I-PER': 2,
        'B-LOC': 3,
        'I-LOC': 4,
        'B-ORG': 5,
        'I-ORG': 6,
        'B-MISC': 7,
        'I-MISC': 8,
    }

    train_taggings = []
    train_sentences = []
    word_embd_weights = None
    word_dict = None
    char_dict = None
    max_word_len = None
    valid_sentences = []
    valid_taggings = []
    valid_steps = None
    model = None
    total_pred, total_true, matched_num = 0, 0, 0.0
    '''
    The paper is for using Parallel Recurrent Neural Networks to do the NER
    '''

    def evaluate(self, predictions, ground_truths, *args, **kwargs):
        eps = 1e-6
        precision = (self.matched_num + eps) / (self.total_pred + eps)
        recall = (self.matched_num + eps) / (self.total_true + eps)
        f1 = 2 * precision * recall / (precision + recall)
        return precision, recall, f1

    def convert_ground_truth(self, data, *args, **kwargs):
        '''
        This method will return the ground truth of the data. 
        The data must be from training dataset e.g. connl2003/train.txt, 
        testing dataset e.g. connl2003/valid.txt or dev dataset e.g.connl2003/test.txt
        For example:
        data = 
         [
        [('EU','NNP','B-NP','B-ORG'),('rejects','VBZ','B-VP','O'),('German','JJ','B-NP','B-MISC'),('call','NN','I-NP','O'),('to','TO','B-VP','O'),('boycott','VB', 'I-VP','O'),('British','JJ', 'B-NP','B-MISC'),('lamb','NN, 'I-NP','O'),('.','.', 'O','O')],
        [('Peter','NNP', 'B-NP','B-PER'),('Blackburn','NNP', 'I-NP','I-PER')]
        ]
        output = convert_ground_truth(data)
        The output should be
        [
        ['B-ORG','O','B-MISC','O','O','O','B-MISC','O','O'],
        ['B-PER','I-PER']
        ]
        :param data: The data is a list. Each element in the list should be a list that contains the tokens for a sentences 
        :return: a list in which a element is another list that contains tags
        :raise: if the data contains a token having no tag
         '''
        pass

    def read_dataset(self, training_files=None, validate_files=None, *args, **kwargs):
        '''
        The method is for reading the data from files.
        For example:
        inputFiles = ['thisDir/file1.txt','thatDir/file2.txt','./file1.txt']
        data = read_dataset(inputFiles)
        data could be 
        [
        [('EU','NNP','B-NP','B-ORG'),('rejects','VBZ','B-VP','O'),('German','JJ','B-NP','B-MISC'),('call','NN','I-NP','O'),('to','TO','B-VP','O'),('boycott','VB', 'I-VP','O'),('British','JJ', 'B-NP','B-MISC'),('lamb','NN, 'I-NP','O'),('.','.', 'O','O')],
        [('Peter','NNP', 'B-NP','B-PER'),('Blackburn','NNP', 'I-NP','I-PER')]
        ]
        This method will do a preprocess for the input files such as embedding the words. The preprocessed data will be stored in a file locally.
        :param training_files: The path of the training data, testing data or validation data
        :return: the list in which each element is a list of token and tag
        :raise: if the file is not accessible or wrong format
        '''
        if validate_files is None:
            validate_files = ['paper2/CoNNL2003eng/valid.txt']
        if training_files is None:
            training_files = ['paper2/CoNNL2003eng/train.txt']
        for path in training_files:
            sentences, taggings = self.load_data(path)
            self.train_sentences.extend(sentences)
            self.train_taggings.extend(taggings)
        for path in validate_files:
            sentences, taggings = self.load_data(path)
            self.valid_sentences.extend(sentences)
            self.valid_taggings.extend(taggings)
        return self.train_sentences, self.train_taggings, self.valid_sentences, self.valid_taggings

    def load_data(self, path):
        sentences, taggings = [], []
        with codecs.open(path, 'r', 'utf8') as reader:
            for line in reader:
                line = line.strip()
                if not line:
                    if not sentences or len(sentences[-1]) > 0:
                        sentences.append([])
                        taggings.append([])
                    continue
                parts = line.split()
                if parts[0] != '-DOCSTART-':
                    sentences[-1].append(parts[0])
                    taggings[-1].append(self.TAGS[parts[-1]])
        if not sentences[-1]:
            sentences.pop()
            taggings.pop()
        return sentences, taggings

    # @classmethod
    def train(self, data=None, *args, **kwargs):
        '''
       This method is for training the dilated cnn model. It will update the model weights. 
        For example:
        data = 
       [
        [('EU','NNP','B-NP','B-ORG'),('rejects','VBZ','B-VP','O'),('German','JJ','B-NP','B-MISC'),('call','NN','I-NP','O'),('to','TO','B-VP','O'),('boycott','VB', 'I-VP','O'),('British','JJ', 'B-NP','B-MISC'),('lamb','NN, 'I-NP','O'),('.','.', 'O','O')],
        [('Peter','NNP', 'B-NP','B-PER'),('Blackburn','NNP', 'I-NP','I-PER')]
        ]
        train(data)
        :param data: a list in which each element is a list containing token w/o tag. If a token is a tuple, the tag will be ignored for training.
        :return: None 
        '''
        dicts_generator = get_dicts_generator(
            word_min_freq=2,
            char_min_freq=2,
            word_ignore_case=True,
            char_ignore_case=False
        )
        for sentence in self.train_sentences:
            dicts_generator(sentence)
        self.word_dict, self.char_dict, self.max_word_len = dicts_generator(return_dict=True)
        if os.path.exists(self.WORD_EMBD_PATH):
            print('Embedding...')
            self.word_dict = {
                '': 0,
                '<UNK>': 1,
            }
            with codecs.open(self.WORD_EMBD_PATH, 'r', 'utf8') as reader:
                print('Embedding open file')
                for line in reader:
                    line = line.strip()
                    if not line:
                        continue
                    word = line.split()[0].lower()
                    if word not in self.word_dict:
                        self.word_dict[word] = len(self.word_dict)
                print('Embedding for loop')
            self.word_embd_weights = get_embedding_weights_from_file(
                self.word_dict,
                self.WORD_EMBD_PATH,
                ignore_case=True,
            )
            print('Embedding done')
        else:
            self.word_embd_weights = None
        print('Embedding all done')
        train_steps = (len(self.train_sentences) + self.BATCH_SIZE - 1) // self.BATCH_SIZE
        valid_steps = (len(self.valid_sentences) + self.BATCH_SIZE - 1) // self.BATCH_SIZE

        self.model = build_model(rnn_num=self.RNN_NUM,
                                 rnn_units=self.RNN_UNITS,
                                 word_dict_len=len(self.word_dict),
                                 char_dict_len=len(self.char_dict),
                                 max_word_len=self.max_word_len,
                                 output_dim=len(self.TAGS),
                                 word_embd_weights=self.word_embd_weights)
        self.model.summary()

        if os.path.exists(self.MODEL_PATH):
            print("loading model from: ", self.MODEL_PATH)
            self.model.load_weights(self.MODEL_PATH, by_name=True)
        else:
            print('Fitting...')
            self.model.fit_generator(
                generator=self.batch_generator(self.train_sentences, self.train_taggings, train_steps),
                steps_per_epoch=train_steps,
                epochs=self.EPOCHS,
                validation_data=self.batch_generator(self.valid_sentences, self.valid_taggings, valid_steps),
                validation_steps=valid_steps,
                callbacks=[
                    keras.callbacks.EarlyStopping(monitor='val_loss', patience=2),
                    keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy', patience=2),
                ],
                verbose=True,
            )

            self.model.save_weights(self.MODEL_PATH)

    # @classmethod
    def predict(self, data, *args, **kwargs):
        '''
        This method is for predicting tags for tokens. The input data is a list in which each element is a list for tokens w/o tags
        For example:
         data = 
        [
        [('EU','NNP','B-NP','B-ORG'),('rejects','VBZ','B-VP','O'),('German','JJ','B-NP','B-MISC'),('call','NN','I-NP','O'),('to','TO','B-VP','O'),('boycott','VB', 'I-VP','O'),('British','JJ', 'B-NP','B-MISC'),('lamb','NN, 'I-NP','O'),('.','.', 'O','O')],
        [('Peter','NNP', 'B-NP','B-PER'),('Blackburn','NNP', 'I-NP','I-PER')]
        ]
        OR
        data = 
              [
        [('EU','NNP','B-NP'),('rejects','VBZ','B-VP'),('German','JJ','B-NP'),('call','NN','I-NP'),('to','TO','B-VP'),('boycott','VB', 'I-VP'),('British','JJ', 'B-NP'),('lamb','NN, 'I-NP'),('.','.', 'O')],
        [('Peter','NNP', 'B-NP','B-PER'),('Blackburn','NNP', 'I-NP','I-PER')]
        ]
        The ouput is a list in which each element is a list of tags for each sentence.
        For example:
         [
        ['B-ORG','O','B-MISC','O','O','O','B-MISC','O','O'],
        ['B-PER','I-PER']
        ]
        :param data: a list in which each element is a list containing tokens w/o tags
        :return: list of tags
        '''
        test_sentences, test_taggings = self.load_data(data)
        test_steps = (len(test_sentences) + self.BATCH_SIZE - 1) // self.BATCH_SIZE
        # eps = 1e-6
        predicts = []
        true_tags = []
        # total_pred, total_true, matched_num = 0, 0, 0.0
        for inputs, batch_taggings in self.batch_generator(test_sentences,test_taggings,test_steps,training=False):
            predict = self.model.predict_on_batch(inputs)
            predict = numpy.argmax(predict, axis=2).tolist()
            for i, pred in enumerate(predict):
                predicts.append((list(map(lambda x: self.fromValueToKey(x), pred[:len(batch_taggings[i])]))))
                true_tags.append((list(map(lambda x: self.fromValueToKey(x), batch_taggings[i]))))
                pred = self.get_tags(pred[:len(batch_taggings[i])])
                true = self.get_tags(batch_taggings[i])
                self.total_pred += len(pred)
                self.total_true += len(true)
                self.matched_num += sum([1 for tag in pred if tag in true])
        return predicts, true_tags

    def fromValueToKey(self, value):
        for tag, index in self.TAGS.items():
            if index == value:
                return tag

    def batch_generator(self, sentences, taggings, steps, training=True):
        while True:
            for i in range(steps):
                batch_sentences = sentences[self.BATCH_SIZE * i:min(self.BATCH_SIZE * (i + 1), len(sentences))]
                batch_taggings = taggings[self.BATCH_SIZE * i:min(self.BATCH_SIZE * (i + 1), len(taggings))]
                word_input, char_input = get_batch_input(
                    batch_sentences,
                    self.max_word_len,
                    self.word_dict,
                    self.char_dict,
                    word_ignore_case=True,
                    char_ignore_case=False
                )
                if not training:
                    yield [word_input, char_input], batch_taggings
                    continue
                sentence_len = word_input.shape[1]
                for j in range(len(batch_taggings)):
                    batch_taggings[j] = batch_taggings[j] + [0] * (sentence_len - len(batch_taggings[j]))
                batch_taggings = self.to_categorical_tensor(numpy.asarray(batch_taggings), len(self.TAGS))
                yield [word_input, char_input], batch_taggings
            if not training:
                break

    def to_categorical_tensor(self, x3d, n_cls):
        batch_size, n_rows = x3d.shape
        x1d = x3d.ravel()
        y1d = keras.utils.to_categorical(x1d, num_classes=n_cls)
        y4d = y1d.reshape([batch_size, n_rows, n_cls])
        return y4d

    def get_tags(self, tags):
        filtered = []
        for i in range(len(tags)):
            if tags[i] == 0:
                continue
            if tags[i] % 2 == 1:
                filtered.append({
                    'begin': i,
                    'end': i,
                    'type': i,
                })
            elif i > 0 and tags[i - 1] == tags[i] - 1:
                filtered[-1]['end'] += 1
        return filtered

