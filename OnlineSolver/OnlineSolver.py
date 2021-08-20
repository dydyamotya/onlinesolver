import configparser
import datetime
import logging
import pathlib
import shutil
import sys
import threading
import time
import typing

import numpy as np
from PySide2 import QtWidgets, QtGui, QtCore
from PySide2.QtWidgets import QFileDialog

import functions
import model


gases_russian_names_map  = {
    "air": "воздух",
    "laurel": "лавр",
    "cinnamon": "корица"
}
cwd = pathlib.Path().cwd()
logs_folder = cwd / "logs"
config_file = cwd / "config.conf"
pictures_folder = cwd / "pictures"
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
        self.data = self.import_data(init_data)  # list

        for i in range(4):
            # noinspection PyTypeChecker
            np.savetxt('temp' + str(i) + '.txt', self.data[i])

        return_flag = True
        return return_flag

    def import_data(self, data):
        return [data[:, i:i + 2] for i in range(7, 14, 2)]

    def get_(self, sens_num):
        return self.data[sens_num]


class CalcThread(threading.Thread):
    def __init__(self, frame):
        super().__init__()
        # There must be the loading of the model
        """В будущем, здесь, скорее всего, будет пять моделей.
        На каждый сенсор по одной, и одна для комплексного определения.
        Современный режим моделей подходит для этой задачи.
        Необходимо будет только каждую сетку раскидать по своим папкам.
        """

        self.frame = frame
        self.daemon = True
        self.stopEvent = threading.Event()
        self.data = Data()
        self.start()

    def run(self):

        model_path = cwd / "model0"
        # Models assignement
        # 0 - concilium net
        # models - sensors nets
        # ==================================================
        self.model0 = model.Model(model_path)
        # ==================================================
        while not self.stopEvent.is_set():
            time.sleep(1)
            if self.frame.file.set_file():
                logger.debug('Got file')
                self.frame.status.setText("Получен новый цикл из файла {}".format(self.frame.file.get_file_name()))
                if self.data.take_data(self.frame.file.read_data()):
                    answers = []
                    answer, _ = self.model0.evaluate(self.data.get_(0))
                    answers.append(answer)
                    self.frame.print_results(answers)

    def stop_thread(self):
        self.stopEvent.set()

    def is_stopped(self):
        return self.stopEvent.is_set()


class PixmapLabel(QtWidgets.QLabel):
    def __init__(self, parent=None):
        super(PixmapLabel, self).__init__(parent=parent)
        self.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        self.setFixedSize(600, 600)
        self.setFrameStyle(QtWidgets.QFrame.Sunken | QtWidgets.QFrame.Panel)
        gases_names = ("air", "laurel", "cinnamon")
        self.gas_name_mapping = dict(
            zip(gases_names, (QtGui.QPixmap((pictures_folder / "{}.jpg".format(name)).as_posix()) for name in gases_names)))

    def choose_picture(self, gas_name):
        self.setPixmap(self.gas_name_mapping[gas_name])

class CustomMainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(CustomMainWindow, self).__init__()
        self.setWindowTitle("OnlineSolver")

        self.file = None
        self.reader_path = None

        self.config = configparser.ConfigParser()
        self.read_config()

        self.worker = None

        self.status = self.statusBar()
        self.status.setText = self.status.showMessage

        self._init_ui()

    def _init_ui(self):

        widget = QtWidgets.QWidget()

        self.setCentralWidget(widget)

        main_layout = QtWidgets.QVBoxLayout(widget)

        buttons_layout = QtWidgets.QHBoxLayout()

        self.reader_path_button = QtWidgets.QPushButton("Выбрать путь до файлов", self)
        self.reader_path_button.clicked.connect(self.on_reader_path_load)

        self.start_button = QtWidgets.QPushButton("Старт", self)
        self.start_button.clicked.connect(self.on_start)
        self.stop_button = QtWidgets.QPushButton("Стоп", self)
        self.stop_button.clicked.connect(self.on_stop)

        buttons_layout.addWidget(self.reader_path_button)
        buttons_layout.addWidget(self.start_button)
        buttons_layout.addWidget(self.stop_button)

        data_layout = QtWidgets.QVBoxLayout()

        self.status1 = QtWidgets.QLabel(self)
        self.status1.setFont(QtGui.QFont("Dejavu Sans", 40))
        self.status1.setAlignment(QtCore.Qt.AlignCenter)
        self.result_widgets = [self.status1]

        data_layout.addWidget(self.status1)

        picture_layout = QtWidgets.QVBoxLayout()

        self.pixmap_label = PixmapLabel(self)

        picture_layout.addWidget(self.pixmap_label)

        main_layout.addLayout(buttons_layout)
        main_layout.addLayout(data_layout)
        main_layout.addLayout(picture_layout)

        self.show()

    def read_config(self):
        logger.debug(config_file.as_posix())
        self.config.read(config_file.as_posix())

    def write_config(self):
        with config_file.open("w") as fd:
            self.config.write(fd)

    def on_start(self):
        if self.worker:
            self.status.setText("Worker already there.")
        if not self.reader_path:
            self.status.setText("Cant start, choose correct temp folder.")
        if not self.worker and self.reader_path:
            self.file = functions.FileReader(self.reader_path)
            self.status.setText("Waiting for results.")
            self.worker = CalcThread(self)

    def on_stop(self):
        try:
            self.worker.stop_thread()
        except AttributeError:
            pass
        else:
            self.worker = None
        finally:
            self.status.setText("Work stopped.")

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.on_stop()
        super(CustomMainWindow, self).closeEvent(event)

    def print_results(self, answers: typing.Iterable):
        for answer, widget in zip(answers, self.result_widgets):
            widget.setText(gases_russian_names_map[answer])

        self.pixmap_label.choose_picture(answers[0])

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
    app = QtWidgets.QApplication()
    main_window = CustomMainWindow()
    sys.exit(app.exec_())
