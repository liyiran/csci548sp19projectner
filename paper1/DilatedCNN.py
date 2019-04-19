from ner import NER


class DilatedCNN(NER):
    '''
    The evaluation method is a common method in the parent class
    This paper is for using dilated cnn to speed up the training speed
    '''

    def evaluate(self, predictions, ground_truths, *args, **kwargs):
        pass

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

    def read_dataset(self, fileNames, *args, **kwargs):
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
        :param fileNames: The path of the training data, testing data or validation data
        :return: the list in which each element is a list of token and tag
        :raise: if the file is not accessible or wrong format
        '''
        
        pass

    @classmethod
    def train(self, data, *args, **kwargs):
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
        pass

    @classmethod
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
        pass
