import logging
import functions
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg
import wx
import threading
import full_learn_package as flp
import main_cut_package as mcp
import numpy as np
import model


ID_START = wx.NewId()
ID_STOP = wx.NewId()
EVT_RESULT_ID =  wx.NewId()
ID_READER = wx.NewId()


def EVT_RESULT(win, func):
    win.Connect(-1, -1, EVT_RESULT_ID, func)

def purify(string):
    return string[:string.find('\n')]

def meaner(array, window = 20):
    size = array.shape[0]
    return np.hstack([np.mean(array[i:i+window]) for i in range(0, size, window)])

def scalling(array):
    array = array.copy()
    min_ = np.min(array)
    array -= min_
    max_ = np.max(array)
    array /= max_
    return array

def smoothing(row, win=5):
    """
    Smooth given row in window with width = win.
    """
    array = np.array(row).ravel()
    new_array = np.empty(array.shape)
    offset1 = win//2
    offset2 = win - offset1
    array_size = len(array)
    for i in range(array_size):
        if i < offset1:
            new_array[i] = np.mean(array[:i+offset2])
        elif i > array_size - offset2:
            new_array[i] = np.mean(array[i-offset1:])
        else:
            new_array[i] = np.mean(array[i-offset1:i+offset2])
    return new_array

class ResultEvent(wx.PyEvent):
    def __init__(self, data):
        super().__init__()
        self.SetEventType(EVT_RESULT_ID)
        self.data = data
    
def reduce_point_number(array: np.array, window: int, shift: int = None) -> np.array:
    """This function reduce the number of points in the scaling maner.
    Cause of that, needed information may be lost.
    Input:      array : 1-D np.array to convert
                window  : int
                shift : int
    Returns:            1-D np.array
    
    Comment: shape = window * steps"""
    data = array.copy()
    data = data.ravel()
    if shift:
        if shift >= window:
            raise Exception("Shift must be less than window")
    else:
        shift = 0
    steps = data.shape[0] // window #aka number of points to return
    to_return = np.empty((steps, ))
    for i in range(steps):
        to_return[i] = data[i * window + shift]
    return to_return

class Data():
    def __init__(self):
        print('Data Initialised')
        self.data = np.array([])
    def take_data(self, init_data):
        new_data = self.import_data(init_data)
        if self.data.shape[0] > new_data.shape[0]:
            self.data = np.vstack([self.data[new_data.shape[0]:], new_data])
        else:
            self.data = np.vstack([np.zeros(new_data.shape), new_data])

        for i in range(4):
            np.savetxt('temp'+str(i)+'.txt', self.data[i])
    def import_data(self, data):
        return [data[:, i:i+1] for i in range(7, 14, 2)]
    def get_(self, sens_num):
        return self.data[sens_num][450:1000].T
        

class CalcThread(threading.Thread):
    def __init__(self, modelPath, frame):
        super().__init__()
        #There must be the loading of the model
        """В будущем, здесь, скорее всего, будет пять моделей.
        На каждый сенсор по одной, и одна для комплексного определения.
        Современный режим моделей подходит для этой задачи.
        Необходимо будет только каждую сетку раскидать по своим папкам.
        """
        model_paths = ["./model{}".format(i) for i in range(5)]
        #Models assignement
        #0 - concilium net
        #models - sensors nets
        #==================================================
        self.model0, *self.models = model.CreateModels(model_paths)
        #==================================================
        self.frame = frame
        self.stopEvent = threading.Event()
        self.data = Data()
        self.start()
    def run(self):
        while not self.stopEvent.is_set():
            if self.frame.file.set_file():
                print('Got file')
                self.data.take_data(self.frame.file.read_data())
                #There model must calculate the answer
                answers = []
                model_vectors = []
                for idx, model_ in enumerate(self.models):
                    answer, vector = model_.Evaluate(self.data.get_(idx))
                    answers.append(answer)
                    model_vectors.append(vector)
                answer, _ = self.model0.Evaluate(np.hstack(model_vectors))
                answers.append(answer)
                self.frame.printResults(answers)

    def stopThread(self):
        self.stopEvent.set()
    def is_stopped(self):
        return self.stopEvent.is_set()


class Window(wx.Frame):
    def __init__(self, parent, title, logger):
        super().__init__(parent, title=title, size=wx.Size(500, 400))

        self.logger = logger
        self.modelPath = self.loadPath()
        self.readerPath = None
        self.worker = None
        self.file = None
        EVT_RESULT(self,self.OnResult)

        self._createGUI()
        self.Show(True)
        
    def loadPath(self):
        with open('config.conf', 'r') as fd:
            temp = fd.readline()
            print(temp)
            while temp != '':
                if temp.startswith('ModelPath::'):
                    break
                temp = fd.readline()
                print(temp)
        return purify(temp.split('::')[1])

    def _createGUI(self):

        

        #Buttons

        self.startButton = wx.Button(self, ID_START, 'Start')
        self.stopButton = wx.Button(self, ID_STOP, 'Stop')
        
        self.loadReaderButton = wx.Button(self, ID_READER, 'Load Reader')
        self.Bind(wx.EVT_BUTTON, self.OnStart, id=ID_START)
        self.Bind(wx.EVT_BUTTON, self.OnStop, id=ID_STOP)
        
        self.Bind(wx.EVT_BUTTON, self.OnReaderLoad, id=ID_READER)

        #Statuses

        self.status1 = wx.TextCtrl(self, -1, '', style=wx.TE_CENTER | wx.TE_READONLY, size = wx.Size(250, 20))
        self.status2 = wx.TextCtrl(self, -1, '', style=wx.TE_CENTER | wx.TE_READONLY, size = wx.Size(250, 20))
        self.status3 = wx.TextCtrl(self, -1, '', style=wx.TE_CENTER | wx.TE_READONLY, size = wx.Size(250, 20))
        self.status4 = wx.TextCtrl(self, -1, '', style=wx.TE_CENTER | wx.TE_READONLY, size = wx.Size(250, 20))
        self.all_status = wx.TextCtrl(self, -1, '', style=wx.TE_CENTER | wx.TE_READONLY, size = wx.Size(250, 20))

        self.readerPathStatus = wx.TextCtrl(self, -1, '...', style = wx.TE_LEFT, size = wx.Size(500, 20))

        #Sizer

        mainSizer = wx.BoxSizer(wx.VERTICAL)
        gasesSizer = wx.FlexGridSizer(5, 2, 2, 2)
        gasesSizer.AddGrowableCol(1)
        buttonSizer = wx.BoxSizer(wx.HORIZONTAL)
        labelSizer = wx.BoxSizer(wx.VERTICAL)

        gasesSizer.Add(wx.StaticText(self, -1, 'Sensor 1'))
        gasesSizer.Add(self.status1, 0, flag = wx.LEFT)
        gasesSizer.Add(wx.StaticText(self, -1, 'Sensor 2'))
        gasesSizer.Add(self.status2, 0, flag = wx.LEFT)
        gasesSizer.Add(wx.StaticText(self, -1, 'Sensor 3'))
        gasesSizer.Add(self.status3, 0, flag = wx.LEFT)
        gasesSizer.Add(wx.StaticText(self, -1, 'Sensor 4'))
        gasesSizer.Add(self.status4, 0, flag = wx.LEFT)
        gasesSizer.Add(wx.StaticText(self, -1, 'Sensors Array'))
        gasesSizer.Add(self.all_status, 0, flag = wx.LEFT)

        buttonSizer.Add(self.startButton, 0, flag = wx.LEFT)
        buttonSizer.Add(self.stopButton, 0, flag = wx.LEFT)
        buttonSizer.Add(self.loadReaderButton, 0, flag = wx.LEFT)

        labelSizer.Add(self.readerPathStatus, 0, flag = wx.LEFT | wx.EXPAND)


        mainSizer.Add(gasesSizer, 0, flag = wx.LEFT | wx.EXPAND)
        mainSizer.Add(buttonSizer, 0, flag = wx.LEFT | wx.EXPAND)
        mainSizer.Add(labelSizer, 0, flag = wx.LEFT | wx.EXPAND)
        self.SetSizer(mainSizer)
    
    def OnStart(self, event):
        if not self.worker and self.readerPath:
            self.all_status.SetLabel('Waiting for results')
            self.worker = CalcThread(self.modelPath, self)
        else:
            self.all_status.SetLabel('Cant start. Choose temp folder.')
    def OnStop(self, event):
        if self.worker:
            self.worker.stopThread()
            self.worker = None
    def OnResult(self, event):
        pass
    def printResults(self, answers : list):
        for answer, widget, index in zip(answers, (self.status1, self.status2, self.status3, self.status4, self.all_status), (1,2,3,4,0)):
            widget.SetLabel(answer)
            self.logger.info(answer, extra={'sensor' : str(index) if index > 0 else 'all'})
    def OnReaderLoad(self, event):
        openDialog = wx.DirDialog(self, 'Choose a temp folder', '', wx.DD_DEFAULT_STYLE | wx.DD_DIR_MUST_EXIST)
        if openDialog.ShowModal() == wx.ID_OK:
            self.readerPath = openDialog.GetPath()
            self.file = functions.FileReader(self.readerPath)
            self.readerPathStatus.SetLabel(self.readerPath)

if __name__ == '__main__':
    #Log init
    FORMAT = "%(asctime)-15s\t%(sensor)s\t%(message)s"
    logging.basicConfig(filename='sample.log', level=logging.INFO, format=FORMAT)
    logger = logging.getLogger('main')
    # ================
    #Main App
    app = wx.App(False)  # Create a new app, don't redirect stdout/stderr to a window.
    wnd = Window(None, 'Electronic Nouse', logger)
    app.MainLoop()

