import pathlib
import pickle

import numpy as np
from tensorflow.keras.models import model_from_json

CLASS_LIST = ["air", "laurel", "cinnamon"]


def bookstein(x, first_idx=1020, last_idx=2040):
    x = np.array(x)
    nj = x.shape[0]
    j = np.ones(nj)
    w = (x[:, 0] + (1j) * x[:, 1] - (j * (x[first_idx, 0] + (1j) * x[first_idx, 1]))) / (
            x[last_idx, 0] + (1j) * x[last_idx, 1] - x[first_idx, 0] - (1j) * x[first_idx, 1]) - 0.5
    w = w[0:nj]
    y = np.real(w)
    z = np.imag(w)
    u = np.vstack([y, z])
    return u


def aggregator(x, wndw=5):
    arr_tmp = np.array([np.median(x[:, i - wndw:i], axis=1) for i in range(wndw, x.shape[1], wndw)]).T
    return arr_tmp


class Model(object):
    """
    Объект класса загружает в память файлы, необходимые для восстановления tensorflow модели.
    После этого соответствующие методы служат для того, чтобы посчитать ответ модели.
    """

    def __init__(self, modelPath: pathlib.Path):
        self.path = modelPath
        self.workable = False
        self._LoadModel()

    def _LoadModel(self):
        """Load the model, reading the needed files in folder"""
        # load scaler model
        with (self.path / 'scaler_x').open('rb') as f:
            self.scaler = pickle.load(f)
            f.close()

        # load neural net model
        with (self.path / 'model.json').open('r') as json_file:
            loaded_model_json = json_file.read()
            self.model = model_from_json(loaded_model_json)
            json_file.close()

        # load model weights
        self.model.load_weights((self.path / 'checkpoint').as_posix())

    def sample_prepare(self, data_sample: np.ndarray):
        raw_data = data_sample
        '''Sample preprocessing
            1st row = T, 2nd row = R

            Step 1: median smooth
            Step 2: bookstein mapping
            Step 3: MinMax scaling
            Step 4: reshape for conv1d input layer'''
        data_smoothed = aggregator(np.array([np.log10(raw_data[1]), raw_data[0] / 50]), 5)
        shape_data = bookstein(data_smoothed.transpose(), 60, 90)[1]
        scaled_sample = self.scaler.transform([shape_data])
        return scaled_sample.reshape([1, 119, 1])

    def Evaluate(self, vector: np.ndarray, threshold: float = 33.3) -> (str, np.ndarray):
        """ Takes the vector to define the answer
            Returns:
                answer : string
                array_to_common_net : np.array
            """
        # model evaluation
        vector = vector.transpose()
        vector = vector[:, :598]
        input_sample = self.sample_prepare(vector)
        pred_index, = self.model.predict_classes(input_sample)
        gas = CLASS_LIST[pred_index]
        return gas, np.array([])


def CreateModels(models_paths):
    return [Model(model_path) for model_path in models_paths]
