
import pandas as pd
import numpy as np
import pickle, os        


def bookstein(x):
    x = np.array(x)
    nj = x.shape[0] #length
    j = np.ones(nj) 
    w = (x[:,0] + (1j)*x[:,1] - (j*(x[0,0] + (1j)*x[0,1])))/(x[1,0]+(1j)*x[1,1] - x[0,0] - (1j)*x[0,1])-0.5
    w = w[0:nj]
    y=np.real(w)
    z = np.imag(w)
    print("I print y-shape", y.shape)
    u = np.vstack([y, z])
    return u


class Model(object):
    
        
    def sample_prepare(self, data_sample):
        
        #print(data_sample)
       
        #to avoid -inf
        def __aggregator(x, wndw=8):
            x = np.array(x)
            arr_tmp = np.array([np.mean(x[:,i-wndw:i], axis=1).reshape([2,1]) for i in range(wndw, x.shape[1], wndw)]).reshape(-1,2)
            arr_tmp= np.concatenate([arr_tmp, np.mean(x[:,-x.shape[1]%wndw:], axis=1).reshape([1,2])], axis=0)
            out = np.transpose(arr_tmp)

            return(out)

        agregated_sample = __aggregator(data_sample)
        agregated_sample[0,:]=agregated_sample[0,:]/100
        agregated_sample[1,:]=np.log10(agregated_sample[1,:])
        
        #print(agregated_sample)
        #print(bookstein( np.transpose(agregated_sample) ))
        r_bookstein = bookstein(np.transpose(agregated_sample))[:, 2:]
        
        bookstein_vec = np.hstack([r_bookstein[0], r_bookstein[1]]).reshape([1,-1])
        
        #print(bookstein_vec)
#         bookstein_vec_scaled = scalerX.transform(bookstein_vec)

        return bookstein_vec

    
    
    
    def classsif_model(self, weights_path):

        from keras.models import Sequential
        from keras.layers.core import Dense, Activation, Dropout
        from keras.layers import BatchNormalization
        
        model = Sequential()
        model.add(Dense(80, input_dim=134,init='uniform', activation='tanh', use_bias=True))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(160, input_dim=80,init='uniform', activation='tanh', use_bias=True))
        model.add(BatchNormalization())
        model.add(Dense(2, input_dim=160,init='uniform', activation='softmax', use_bias=True))
        model.compile(loss='categorical_crossentropy', optimizer='adagrad', metrics=['accuracy'])
        model.load_weights(weights_path)
        return model
    
    
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
        self.scaler_gases = None
        self.scaler_air = None
        self.model_air = None
        self.model_gases = None
        for file in list(os.listdir(self.path)):
            if 'scaler_air' in file:
                with open(self.path+'/'+file, 'rb') as fd:
                    self.scaler_air = pickle.load(fd)
            elif 'scaler_gases' in file:
                with open(self.path+'/'+file, 'rb') as fd:
                    self.scaler_gases = pickle.load(fd)
            elif 'weights_air' in file:
                self.model_air = self.classsif_model(self.path+'/'+file)
            elif 'weights_gases' in file:
                self.model_gases = self.classsif_model(self.path+'/'+file)
#             elif 'weights_regression_h2' in file:
#                 self.weights_regression_h2 = self.path+'/'+file
#             elif 'weights_regression_propane' in file:
#                 self.weights_regression_propane = self.path+'/'+file  
                    
        self._CheckFiles()

    def _CheckFiles(self):
        if (self.scaler_gases and self.scaler_air and self.model_air and self.model_gases):
            self.workable = True

    def Evaluate(self, vector : np.array, threshold : float = 33.3) -> (str, np.array):
        """ Takes the vector to define the answer
            Returns:
                answer : string
                array_to_common_net : np.array
            """
        if not self.workable:
            return "No model", np.zeros(3)

        vector = self.sample_prepare(vector)
        
        vector_air = self.scaler_air.transform(vector)
        vector_gases = self.scaler_gases.transform(vector)
        
        
        answer_air = list(self.model_air.predict(vector_air)[0]) #ndarray
        print('air predict; air = [0,1]', answer_air)
        answer_gases = [0,0]
        if answer_air[0]>0.5:
            answer_gases = list(self.model_gases.predict(vector_gases)[0])
            print('gases; propane = [0,1]', answer_gases)
            
        answer_gases.append(answer_air[0]) 
        answer_vector = np.array(answer_gases)
        print(answer_vector)
        gas_index = answer_vector.argmax()
        #There must be more smart gas namer.
        #But on the first time, its normal.
        if gas_index == 2: return "Air", answer_vector
        elif gas_index == 0: return "Hydrogen", answer_vector
        else: return "Propane", answer_vector

#     def _EvaluateNet(self, vector : np.array) -> np.array:
#         """Evaluate answer for whole net"""
        
                
#         i = 0
#         for layer in self.config_file:
#             if layer['class_name'] == 'Dropout':
#               continue
#             if layer['config']['activation'] == 'sigmoid':
#               act_func = sigmoid
#             elif layer['config']['activation'] == 'softmax':
#               act_func = softmax
#             elif layer['config']['activation'] == 'linear':
#               act_func = linear
#             elif layer['config']['activation'] == 'tanh':
#               act_func = np.tanh
#             else:
#               raise Exception("Unknown activation function")
#             weights = self.layers_list[i*2]
#             biases = self.layers_list[i*2+1]
#             vector = self._EvaluateLayer(vector, weights, biases, act_func)
#             i += 1
#         return vector

#     def _EvaluateLayer(input_vector, weights, biases, act_func):
#         """Evaluate the answer from one layer"""
#         return act_func(np.matmul(input_vector, weights) + biases)

def CreateModels(models_paths):
    return [Model(model_path) for model_path in models_paths]
