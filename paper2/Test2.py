import unittest

from paper2 import UOI

'''
# Sample workflow:
file_dict = {
                "train": {"data" : "/home/sample_train.txt"},
                "dev": {"data" : "/home/sample_dev.txt"},
                "test": {"data" : "/home/sample_test.txt"},
             }
dataset_name = 'CONLL2003'

# instantiate the class
myModel = myClass()

# read in a dataset for training
data = myModel.read_dataset(file_dict, dataset_name)  
myModel.train(data)  # trains the model and stores model state in object properties or similar

predictions = myModel.predict(data['test'])  # generate predictions! output format will be same for everyone
test_labels = myModel.convert_ground_truth(data['test'])  <-- need ground truth labels need to be in same format as predictions!

P,R,F1 = myModel.evaluate(predictions, test_labels)  # calculate Precision, Recall, F1
print('Precision: %s, Recall: %s, F1: %s'%(P,R,F1))
'''


class Test2(unittest.TestCase):
    uoi = None
    training_sentence = None
    training_tag = None
    validate_sentence = None
    validate_tag = None

    @classmethod
    def setUpClass(cls):
        cls.uoi = UOI.UOI()
        print("set up class")

    def read_dataset(self):
        print("begin test read dataset")
        trainingFile = ["/Users/liyiran/csci548sp19projectner_my/paper2/CoNNL2003eng/train.txt"]
        validFile = ['/Users/liyiran/csci548sp19projectner_my/paper2/CoNNL2003eng/valid.txt']
        self.uoi.read_dataset(trainingFile, validFile)
        print("test read dataset")

    def train(self):
        print("begin test train")
        self.uoi.train(data=None)
        print("test train")

    def predict(self):
        predicts = self.uoi.predict('/Users/liyiran/csci548sp19projectner_my/paper2/CoNNL2003eng/test.txt')
        print(predicts)
        print("test predict")

    def convert_ground_truth(self):
        self.uoi.convert_ground_truth(data=['Nadim'])
        print("test convert to ground truth")

    def evaluation(self):
        self.uoi.evaluate("")
        print("test evaluation")

    def test(self):
        self.read_dataset()
        self.train()
        self.predict()
        self.convert_ground_truth()
        self.evaluation()
