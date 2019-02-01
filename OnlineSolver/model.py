import numpy as np
import pickle
import os


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    x = x.ravel()
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def sigmoid(x: np.array) -> np.array:
    """Compute the sigmoid function"""
    return 1. / (1. + np.exp(-x))


def linear(x):
    return x

class Model(object):
    """
    Объект класса загружает в память файлы, необходимые для восстановления tensorflow модели.
    После этого соответствующие методы служат для того, чтобы посчитать ответ модели.
    """
    def __init__(self, modelPath):
        self.path = modelPath
        self.workable = False
        self._LoadModel()

    def _LoadModel(self):
        """Load the model, reading the needed files in folder"""
        self.config_file = None
        self.layers_list = None
        for file in list(os.listdir(self.path)):
            if 'config' in file:
                with open(self.path+'/'+file, 'rb') as fd:
                    self.config_file = pickle.load(fd)['layers']
            elif 'weights' in file:
                with open(self.path+'/'+file, 'rb') as fd:
                    self.layers_list = pickle.load(fd)
        self._CheckFiles()

    def _CheckFiles(self):
        if (self.config_file and self.layers_list):
            self.workable = True

    def Evaluate(self, vector : np.array, threshold : float = 33.3) -> (str, np.array):
        """ Takes the vector to define the answer
            Returns:
                answer : string
                array_to_common_net : np.array
            """
        if not self.workable:
            return "No model", np.zeros(3)
        answer_vector = self._EvaluateNet(vector)
        if np.max(answer_vector) < threshold:
            return "Undefined Gas", answer_vector
        gas_index = answer_vector.argmax()
        #There must be more smart gas namer.
        #But on the first time, its normal.
        if gas_index == 0: return "Air", answer_vector
        elif gas_index == 1: return "Methane", answer_vector
        else: return "Propane", answer_vector

    def _EvaluateNet(self, vector : np.array) -> np.array:
        """Evaluate answer for whole net"""
        i = 0
        for layer in self.config_file:
            if layer['class_name'] == 'Dropout':
            	continue
            if layer['config']['activation'] == 'sigmoid':
            	act_func = sigmoid
            elif layer['config']['activation'] == 'softmax':
            	act_func = softmax
            elif layer['config']['activation'] == 'linear':
            	act_func = linear
            elif layer['config']['activation'] == 'tanh':
            	act_func = np.tanh
            else:
            	raise Exception("Unknown activation function")
            weights = self.layers_list[i*2]
            biases = self.layers_list[i*2+1]
            vector = self._EvaluateLayer(vector, weights, biases, act_func)
            i += 1
        return vector

    def _EvaluateLayer(input_vector, weights, biases, act_func):
        """Evaluate the answer from one layer"""
        return act_func(np.matmul(input_vector, weights) + biases)

def CreateModels(models_paths):
    return [Model(model_path) for model_path in models_paths]
        
