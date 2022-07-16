import pathlib

from tensorflow.keras.models import load_model
import logging
import yaml
import pandas as pd
import typing
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# CLASS_LIST = ["air", "basil", "jasmin"]

params = yaml.safe_load(open("params.yaml"))["featurize"]


def bookstein(x, first_idx=1020, last_idx=2040):
    x = np.array(x)
    nj = x.shape[0]
    j = np.ones(nj)
    w = (x[:, 0] + (1j) * x[:, 1] -
         (j * (x[first_idx, 0] +
               (1j) * x[first_idx, 1]))) / (x[last_idx, 0] +
                                            (1j) * x[last_idx, 1] -
                                            x[first_idx, 0] -
                                            (1j) * x[first_idx, 1]) - 0.5
    w = w[0:nj]
    y = np.real(w)
    z = np.imag(w)
    u = np.vstack([y, z])
    return u


def aggregator(x, wndw=5):
    arr_tmp = np.array([
        np.median(x[:, i - wndw:i], axis=1)
        for i in range(wndw, x.shape[1], wndw)
    ]).T
    return arr_tmp


def _bookstein_preprocessing(r_data, t_data):
    for (_, r), (_, t) in zip(r_data.iterrows(), t_data.iterrows()):
        data_smoothed = aggregator(
            np.array([np.log10(r.values), t.values / 50]), 5)
        shape_data = bookstein(data_smoothed.transpose(), 0, -1)[1]
        yield shape_data


def bookstein_preprocessing(r_data, t_data):
    return pd.DataFrame(_bookstein_preprocessing(r_data, t_data))


def scale_data(X,
               scaler=None) -> (np.ndarray, typing.Optional[StandardScaler]):
    if scaler is None:
        scaler = StandardScaler()
        new_X = scaler.fit_transform(X)
        return new_X, scaler
    else:
        return scaler.transform(X), scaler


def get_df(data):
    return pd.read_csv(data, sep="\t", header=0)


def separate_to_blocks(data):
    r_columns = tuple(f"r{idx}" for idx in range(601))
    t_columns = tuple(f"t{idx}" for idx in range(601))
    y_columns = ("h2_conc", "h2s_conc")
    return data.loc[:, r_columns], data.loc[:, t_columns], data.loc[:,
                                                           y_columns]


def get_window(r_data, t_data, window_size=None, window_shift=0):
    data_size = r_data.shape[1]
    if window_size is None:
        window_size = data_size
    if (window_size + window_shift) > data_size:
        raise Exception(f"Wrong window size and shift. Data size: {data_size},\
                window size: {window_size}, window shift: {window_shift}")
    return (r_data.iloc[:,
            slice(window_shift, window_shift + window_size, 1)],
            t_data.iloc[:,
            slice(window_shift, window_shift + window_size, 1)])


def u_to_r(u, r4=1.11e7):
    rs1 = 3.00562
    rs2 = 1.991563
    k = 4.068
    return (rs1 - rs2) * r4 / ((2.5 + 2.5 * k - u) / k - rs2) - r4


def r_to_u(r, r4=1.11e7):
    rs1 = 3.00562
    rs2 = 1.991563
    k = 4.068
    return 2.5 + 2.5 * k - k * (rs2 + (rs1 - rs2) * r4 / (r + r4))


def bit_depth_decrease(r_data, aim_bit_depth, r4):
    def bit_depth_decrease_row(row):
        us = r_to_u(row, r4=r4)
        u_step = 5 / (2 ** aim_bit_depth)
        return u_to_r((us // u_step) * u_step, r4=r4)

    return r_data.apply(bit_depth_decrease_row, axis=1)


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
        with (self.path / 'StandartScaler.pkl').open('rb') as f:
            self.scaler = pickle.load(f)

        self.model = load_model(self.path / "motya_model.h5")

    def sample_prepare(self, data_sample: np.ndarray):
        if data_sample.shape[1] == 600:
            print("Shape is equal to 600 suddenly")
            template = np.zeros((2, 601))
            template[:, :600] = data_sample
            template[:, 600] = data_sample[:, 599]
            data_sample = template

        r = pd.DataFrame([data_sample[1, :], ])
        t = pd.DataFrame([data_sample[0, :], ])
        '''Sample preprocessing
            1st row = T, 2nd row = R

            Step 1: median smooth
            Step 2: bookstein mapping
            Step 3: MinMax scaling
            Step 4: reshape for conv1d input layer'''

        if (params["bit-depth"] is not None) and (params["r4"] is not None):
            r = bit_depth_decrease(r, params["bit-depth"], params["r4"])

        # Window getting
        if params["window-size"] is not None:
            r, t = get_window(r,
                              t,
                              window_size=params["window-size"],
                              window_shift=params["window-shift"])

        # Booksteining
        x = bookstein_preprocessing(r, t)

        # Scaling


        x = pd.DataFrame(self.scaler.transform(x.values))

        return x

    def evaluate(self, vector: np.ndarray, threshold: float = 33.3) -> (str, np.ndarray):
        """ Takes the vector to define the answer
            Returns:
                answer : string
                array_to_common_net : np.array
            """
        # model evaluation
        vector = vector.transpose()
        input_sample = self.sample_prepare(vector)
        prediction, = self.model.predict(input_sample)
        logger.info(str(prediction))
        pred_conc = prediction[0]
        # pred_index = filter_output(prediction)
        # gas = CLASS_LIST[pred_index]
        return pred_conc, np.array([])


def create_models(models_paths):
    return [Model(model_path) for model_path in models_paths]
