import configparser
import datetime
import logging
import pathlib
import shutil
import sys
import threading
import typing

import PySide6
import numpy as np
from PySide6 import QtWidgets, QtGui
from PySide6.QtWidgets import QFileDialog

import functions
import model

cwd = pathlib.Path().cwd()
logs_folder = cwd / "logs"
config_file = cwd / "config.conf"
logs_folder.mkdir(exist_ok=True)
logging_file_name = (logs_folder / datetime.datetime.now().strftime("%y%m%d_%H%M%S")).with_suffix(".log")
logging.basicConfig(filename=logging_file_name.as_posix(),
                    filemode='w',
                    level=logging.DEBUG,
                    datefmt="%y%m%d_%H:%M:%S",
                    format='%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger()


def purify(string):
    return string[:string.find('\n')]


class Data():
    def __init__(self):
        logger.debug('Data Initialised')
        self.data = [np.array([])] * 4

    def take_data(self, init_data) -> bool:
        """Returns True, if data was normally taken
        False, if model must wait"""
        return_flag = False
        new_data = self.import_data(init_data)  # list
        logger.debug("New data shape: {}".format(new_data[0].shape))
        for i in range(4):
            if self.data[i].shape[0] > new_data[i].shape[0]:
                if i == 0:
                    logger.debug("Return flag to True")
                    return_flag = True
                self.data[i] = np.vstack([self.data[i][new_data[i].shape[0]:], new_data[i]])
            else:
                self.data[i] = np.vstack([np.zeros(new_data[i].shape), new_data[i]])

        for i in range(4):
            # noinspection PyTypeChecker
            np.savetxt('temp' + str(i) + '.txt', self.data[i])
        return return_flag

    def import_data(self, data):
        return [data[:, i:i + 2] for i in range(7, 14, 2)]

    def get_(self, sens_num):
        return self.data[sens_num][450:1000].T


class CalcThread(threading.Thread):
    def __init__(self, frame):
        super().__init__()
        # There must be the loading of the model
        """В будущем, здесь, скорее всего, будет пять моделей.
        На каждый сенсор по одной, и одна для комплексного определения.
        Современный режим моделей подходит для этой задачи.
        Необходимо будет только каждую сетку раскидать по своим папкам.
        """

        model_paths = [cwd / "model0"]
        # Models assignement
        # 0 - concilium net
        # models - sensors nets
        # ==================================================
        self.model0: model.Model = model.CreateModels(model_paths)
        # ==================================================
        self.frame = frame
        self.stopEvent = threading.Event()
        self.data = Data()
        self.start()

    def run(self):
        while not self.stopEvent.is_set():
            if self.frame.file.set_file():
                logger.debug('Got file')
                if self.data.take_data(self.frame.file.read_data()):
                    answers = []

                    # model_vectors = []
                    # for idx, model_ in enumerate(self.models):
                    #     answer, vector = model_.Evaluate(self.data.get_(idx))
                    #     answers.append(answer)
                    #     model_vectors.append(vector)
                    # answer, _ = self.model0.Evaluate(np.hstack(model_vectors))
                    # self.frame.printResults(answers)

                    answer, vector = self.model0.Evaluate(self.data.get_(0))
                    answers.append(answer)
                    self.frame.print_results(answers)

    def stopThread(self):
        self.stopEvent.set()

    def is_stopped(self):
        return self.stopEvent.is_set()


class CustomMainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(CustomMainWindow, self).__init__()
        self.setWindowTitle("OnlineSolver")

        self.file = None
        self.reader_path = None

        self.config = configparser.ConfigParser()
        self.config.read(config_file)

        self.worker: typing.Optional[CalcThread] = None

        self.status = self.statusBar()
        self.status.setText = self.status.showMessage

        self._init_ui()

    def _init_ui(self):

        widget = QtWidgets.QWidget()

        self.setCentralWidget(widget)

        main_layout = QtWidgets.QVBoxLayout(widget)

        buttons_layout = QtWidgets.QHBoxLayout(self)

        self.reader_path_button = QtWidgets.QPushButton("Choose reader path", self)
        self.reader_path_button.clicked.connect(self.on_reader_path_load)

        self.start_button = QtWidgets.QPushButton("Start", self)
        self.start_button.clicked.connect(self.on_start)
        self.stop_button = QtWidgets.QPushButton("Stop", self)
        self.stop_button.clicked.connect(self.on_stop)

        buttons_layout.addWidget(self.reader_path_button)
        buttons_layout.addWidget(self.start_button)
        buttons_layout.addWidget(self.stop_button)

        data_layout = QtWidgets.QFormLayout(self)

        self.status1 = QtWidgets.QLineEdit(self)
        self.status1.setDisabled(True)
        self.result_widgets = [self.status1]

        data_layout.addRow("Flavor", self.status1)

        main_layout.addLayout(buttons_layout)
        main_layout.addLayout(data_layout)

        self.show()

    def read_config(self):
        self.config.read_file(config_file.as_posix())

    def write_config(self):
        with config_file.open("w") as fd:
            self.config.write(fd)

    def on_start(self):
        start_time: str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if not self.worker and self.reader_path:
            self.file = functions.FileReader(self.reader_path)
            self.status.setText("Waiting for results.")
            self.copy_files_on_start(start_time)
            self.worker = CalcThread(self)
        if self.worker:
            self.status.setText("Worker already there.")
        if not self.reader_path:
            self.status.setText("Cant start, choose correct temp folder.")

    def on_stop(self):
        try:
            self.worker.stopThread()
        except AttributeError:
            pass

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.on_stop()
        super(CustomMainWindow, self).closeEvent(event)

    def print_results(self, answers: typing.Iterable):
        for answer, widget in zip(answers, self.result_widgets):
            widget.setText(answer)

    def on_reader_path_load(self):
        default_dict = self.config["DEFAULT"]
        reader_path = default_dict.get("reader_path", cwd.as_posix())
        dir_name = QFileDialog.getExistingDirectory(parent=self,
                                                    caption="Choose temp path",
                                                    dir=reader_path)
        if dir_name:
            self.config["DEFAULT"] = {"reader_path": dir_name}
            self.write_config()
            self.reader_path = dir_name

    def copy_files_on_start(self, start_time: str):
        reader_path_pathlib = pathlib.Path(self.reader_path)
        files_list = reader_path_pathlib.iterdir()
        new_folder_name = reader_path_pathlib.parent / start_time
        try:
            new_folder_name.mkdir()
        except FileNotFoundError:
            self.status.setText("Wrong file path")
        else:
            for file in files_list:
                shutil.move(file, new_folder_name / file.name)


if __name__ == '__main__':
    # Main App
    app = PySide6.QtWidgets.QApplication()
    main_window = CustomMainWindow()
    sys.exit(app.exec_())
