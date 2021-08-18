import os
import time

import numpy as np


class FileReader:
    def __init__(self, path):
        self.data = None
        self.file = None
        if os.path.exists(path):
            self.path = path
        else:
            raise Exception('Wrong path')

    def check_files(self):
        if len(os.listdir(self.path)) > 1:
            return True
        else:
            return False

    def choose_first(self):
        file_, time_ = None, time.time() + 10
        for file in os.listdir(self.path):
            time_m = os.path.getmtime(self.path + '/' + file)
            if time_ > time_m:
                time_ = time_m
                file_ = self.path + '/' + file
        return file_

    def set_file(self):
        if self.check_files():
            time.sleep(2)
            self.file = self.choose_first()
            return True
        else:
            return False

    def read_data(self):
        with open(self.file, 'rb') as fd:
            self.data = np.fromfile(fd, offset=4, dtype=np.float32).reshape(-1, 15)
            np.savetxt('temp.txt', self.data)
        if os.path.exists(self.file):
            os.remove(self.file)
            self.file = None
        return self.data

    def get_data(self):
        return self.data
