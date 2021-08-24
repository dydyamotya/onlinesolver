import pathlib
import pickle

import numpy as np
from tensorflow.keras.models import model_from_json
from tensorflow.keras.layers import Dense
import tensorflow as tf
import logging
logger = logging.getLogger()


SASHA_CLASS_LIST = ["air", "laurel", "cinnamon"]
CLASS_LIST = ["cinnamon", "laurel", "air"]


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

def filter_output(vector):
    """Classes: air, laurel, cinnamon"""
    a, l, c = vector
    if c > 0.4:
        return 2
    else:
        if c > 0.02:
            return 1
        else:
            return 0

class Model(object):
    """
    Объект класса загружает в память файлы, необходимые для восстановления tensorflow модели.
    После этого соответствующие методы служат для того, чтобы посчитать ответ модели.
    """

    def __init__(self, modelPath: pathlib.Path):
        self.path = modelPath
        self.workable = False
        self._load_model()

    def _load_model(self):
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

        with (self.path / "weights.pkl").open('rb') as fd:
            weights = pickle.load(fd)
            self.model.set_weights(weights)
        # self.model.load_weights((self.path / 'checkpoint').as_posix())

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

    def evaluate(self, vector: np.ndarray, threshold: float = 33.3) -> (str, np.ndarray):
        """ Takes the vector to define the answer
            Returns:
                answer : string
                array_to_common_net : np.array
            """
        # model evaluation
        vector = vector.transpose()
        vector = vector[:, :598]
        input_sample = self.sample_prepare(vector)
        prediction = self.model.predict(input_sample)
        logger.info(str(prediction))
        # pred_index, = np.argmax(prediction, axis=-1)
        pred_index = filter_output(prediction)
        gas = SASHA_CLASS_LIST[pred_index]
        return gas, np.array([])


def create_models(models_paths):
    return [Model(model_path) for model_path in models_paths]
