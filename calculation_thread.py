import threading
import logging
import numpy as np
import time
import model


logger = logging.getLogger(__name__)

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
    def __init__(self, frame, cwd, model_name):
        super().__init__()
        # There must be the loading of the model
        """В будущем, здесь, скорее всего, будет пять моделей.
        На каждый сенсор по одной, и одна для комплексного определения.
        Современный режим моделей подходит для этой задачи.
        Необходимо будет только каждую сетку раскидать по своим папкам.
        """

        self.frame = frame
        self.cwd = cwd
        self.model_name = model_name
        self.daemon = True
        self.stopEvent = threading.Event()
        self.data = Data()
        self.start()

    def run(self):

        model_path = self.cwd / self.model_name
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