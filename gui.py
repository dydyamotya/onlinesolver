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
from calculation_thread import CalcThread
from modbus_support import ModbusThread

gases_russian_names_map  = {
    "air": "воздух",
    "basil": "базилик",
    "jasmin": "жасмин"
}

logger = logging.getLogger(__name__)

def purify(string):
    return string[:string.find('\n')]


class PixmapLabel(QtWidgets.QLabel):
    def __init__(self, pictures_folder, parent=None):
        super(PixmapLabel, self).__init__(parent=parent)
        self.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        self.setFixedSize(600, 600)
        self.setFrameStyle(QtWidgets.QFrame.Sunken | QtWidgets.QFrame.Panel)
        gases_names = ("air", "basil", "jasmin")
        self.gas_name_mapping = dict(
            zip(gases_names, (QtGui.QPixmap((pictures_folder / "{}.jpg".format(name)).as_posix()) for name in gases_names)))

    def choose_picture(self, gas_name):
        self.setPixmap(self.gas_name_mapping[gas_name])

class CustomMainWindow(QtWidgets.QMainWindow):
    def __init__(self, cwd, config_file, modbus_support: bool = False, is_class: bool = True, debug: bool = False, pictures_folder = None):
        super(CustomMainWindow, self).__init__()
        self.setWindowTitle("OnlineSolver")

        self.file = None
        self.reader_path = None

        self.cwd = cwd
        self.config_file = config_file

        self.config = configparser.ConfigParser()
        self.read_config()

        self.worker = None
        self.modbus_worker = None

        self.modbus_support = modbus_support
        self.is_class = is_class
        self.debug = debug
        self.pictures_folder = pictures_folder

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

        neuro_buttons_with_label_layout = QtWidgets.QVBoxLayout()

        neuro_buttons_with_label_layout.addWidget(QtWidgets.QLabel("Нейросеть", self))

        neuro_buttons_layout = QtWidgets.QHBoxLayout()


        self.start_button = QtWidgets.QPushButton("Старт", self)
        self.start_button.clicked.connect(self.on_start)
        self.stop_button = QtWidgets.QPushButton("Стоп", self)
        self.stop_button.clicked.connect(self.on_stop)

        neuro_buttons_layout.addWidget(self.start_button)
        neuro_buttons_layout.addWidget(self.stop_button)

        neuro_buttons_with_label_layout.addLayout(neuro_buttons_layout)

        buttons_layout.addWidget(self.reader_path_button)
        buttons_layout.addLayout(neuro_buttons_with_label_layout)

        if self.modbus_support:
            modbus_buttons_with_label_layout = QtWidgets.QVBoxLayout()

            modbus_label_lineedit_layout = QtWidgets.QHBoxLayout()

            modbus_label_lineedit_layout.addWidget(QtWidgets.QLabel("Газ", self))
            self.modbus_port_lineedit = QtWidgets.QLineEdit(self)
            modbus_label_lineedit_layout.addWidget(self.modbus_port_lineedit)

            modbus_buttons_with_label_layout.addLayout(modbus_label_lineedit_layout)


            modbus_buttons_layout = QtWidgets.QHBoxLayout()

            self.start_modbus_button = QtWidgets.QPushButton("Старт", self)
            self.start_modbus_button.clicked.connect(self.on_modbus_start)
            self.stop_modbus_button = QtWidgets.QPushButton("Стоп", self)
            self.stop_modbus_button.clicked.connect(self.on_modbus_stop)

            modbus_buttons_layout.addWidget(self.start_modbus_button)
            modbus_buttons_layout.addWidget(self.stop_modbus_button)

            modbus_buttons_with_label_layout.addLayout(modbus_buttons_layout)
            buttons_layout.addLayout(modbus_buttons_with_label_layout)

        data_layout = QtWidgets.QVBoxLayout()

        stati_layout = QtWidgets.QHBoxLayout()

        self.status1 = QtWidgets.QLabel(self)
        self.status1.setFont(QtGui.QFont("Dejavu Sans", 40))
        self.status1.setAlignment(QtCore.Qt.AlignCenter)
        self.result_widgets = [self.status1]

        stati_layout.addWidget(self.status1)

        self.gasstatus = QtWidgets.QLabel(self)
        self.gasstatus.setFont(QtGui.QFont("Dejavu Sans", 40))
        self.gasstatus.setAlignment(QtCore.Qt.AlignCenter)
        self.gasstatus.setFixedWidth(50)

        stati_layout.addWidget(self.gasstatus)

        data_layout.addLayout(stati_layout)

        main_layout.addLayout(buttons_layout)
        main_layout.addLayout(data_layout)

        if self.is_class:
            picture_layout = QtWidgets.QVBoxLayout()
            self.pixmap_label = PixmapLabel(self.pictures_folder, self)
            picture_layout.addWidget(self.pixmap_label)
            main_layout.addLayout(picture_layout)

        self.show()

    def read_config(self):
        logger.debug(self.config_file.as_posix())
        self.config.read(self.config_file.as_posix())

    def write_config(self):
        with self.config_file.open("w") as fd:
            self.config.write(fd)

    def on_start(self):
        if self.worker:
            self.status.setText("Worker already there.")
        if not self.reader_path:
            self.status.setText("Cant start, choose correct temp folder.")
        if not self.worker and self.reader_path:
            if not self.debug:
                self.copy_files_on_start()
            self.file = functions.FileReader(self.reader_path)
            self.status.setText("Waiting for results.")
            self.worker = CalcThread(self, self.cwd, "model0")

    def on_stop(self):
        try:
            self.worker.stop_thread()
        except AttributeError:
            pass
        else:
            self.worker = None
        finally:
            self.status.setText("Work stopped.")

    def on_modbus_start(self):
        if self.modbus_worker:
            self.status.setText("Modbus worker already there.")
        if not self.modbus_worker:
            self.modbus_worker = ModbusThread(self.modbus_port_lineedit.text(), self)

    def on_modbus_stop(self):
        try:
            self.modbus_worker.stop_thread()
        except AttributeError:
            pass
        else:
            self.modbus_worker = None


    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.on_stop()
        super(CustomMainWindow, self).closeEvent(event)

    def print_results(self, answers: typing.List):
        if self.is_class:
            for answer, widget in zip(answers, self.result_widgets):
                widget.setText(gases_russian_names_map[answer])
            self.pixmap_label.choose_picture(answers[0])
        else:
            for answer, widget in zip(answers, self.result_widgets):
                widget.setText(str(answer))

    def on_reader_path_load(self):
        default_dict = self.config["DEFAULT"]
        reader_path = default_dict.get("reader_path", self.cwd.as_posix())
        dir_name = QFileDialog.getExistingDirectory(parent=self,
                                                    caption="Choose temp path",
                                                    dir=reader_path)
        if dir_name:
            self.config["DEFAULT"] = {"reader_path": dir_name}
            self.write_config()
            self.reader_path = dir_name

    def copy_files_on_start(self):
        reader_path_pathlib = pathlib.Path(self.reader_path)
        files_list = reader_path_pathlib.iterdir()
        start_time = datetime.datetime.now().isoformat()
        new_folder_name = reader_path_pathlib.parent / start_time
        try:
            new_folder_name.mkdir()
        except FileNotFoundError:
            self.status.setText("Wrong file path")
        else:
            for file in files_list:
                shutil.move(file, new_folder_name / file.name)

    def notice_stop_modbus(self, message):
        self.status.setText(message)
        self.modbus_worker = None

    def print_gasstatus(self, message):
        self.gasstatus.setText(message)

def main(cwd, config_file, modbus_support: bool = False, is_class: bool = True, debug: bool = False, pictures_folder=None):
    if not is_class and pictures_folder is None:
        raise Exception("For isclass == True please specify a pictures folder")
    app = QtWidgets.QApplication()
    main_window = CustomMainWindow(cwd, config_file, modbus_support=modbus_support, is_class=is_class, debug=debug, pictures_folder=pictures_folder)
    sys.exit(app.exec_())
