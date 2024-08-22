import os
import numpy as np
import tensorflow as tf
from tensorflow.python.lib.io import file_io
from google.cloud import storage
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import Sequential
from tensorflow.keras import regularizers
from sklearn.metrics import mean_squared_error
#from tensorflow.keras.losses import MeanSquaredError

class GenericModel:
    def __init__(self,modelConfig):
        self.modelConfig = modelConfig
        self.model=None
        self.train_err=None
        self.validation_err=None
        self.test_err=None
        self.threshold=0.5
        
    def train(self):
        f=file_io.FileIO(os.environ["AIP_TRAINING_DATA_URI"], 'r')
        data=np.genfromtxt(f,delimiter=",")
        m,n=data.shape
        x_data,y_data=np.split(data,[n-1],axis=1)
        x_data_n=self.normalize(x_data)

        self.model = Sequential()
        self.model.add(Input(shape=(n-1,)))
        for index in range(len(self.modelConfig.layers)):
            dense=Dense(self.modelConfig.neurons[index],self.modelConfig.layers[index],
                        kernel_regularizer=regularizers.l2(self.modelConfig.regularizers[index]))
            self.model.add(dense)
        self.model.summary()
        self.model.compile(
            loss=self.modelConfig.loss,
            optimizer=self.modelConfig.optimizer
        )
        self.model.fit(
            x_data_n,y_data,            
            epochs=self.modelConfig.epochs,
            verbose=self.modelConfig.verbose
        ) 
        self.train_err = self.getErr(x_data_n,y_data)
        """
        if (self.modelConfig.model_type=="regression"):
            yhat = self.model.predict(x_data_n)
            self.train_err = mean_squared_error(y_data, yhat) / 2
        if (self.modelConfig.model_type=="classification"):
            yhat = self.model.predict(x_data_n)
            yhat = np.where(yhat >= self.threshold, 1, 0)
            self.train_err = np.mean(yhat != y_data)
        """
    def save(self):
        output_directory = os.environ['AIP_MODEL_DIR']
        self.model.save(output_directory)

    def validate(self):
        f=file_io.FileIO(os.environ["AIP_VALIDATION_DATA_URI"], 'r')
        data=np.genfromtxt(f,delimiter=",")
        m,n=data.shape
        x_data,y_data=np.split(data,[n-1],axis=1)
        x_data_n=self.normalize(x_data)
        self.validation_err = self.getErr(x_data_n,y_data)
        """
        if (self.modelConfig.model_type=="regression"):
            yhat = self.model.predict(x_data_n)
            self.validation_err = mean_squared_error(y_data, yhat) / 2
        if (self.modelConfig.model_type=="classification"):
            yhat = self.model.predict(x_data_n)
            yhat = np.where(yhat >= self.threshold, 1, 0)
            self.validation_err = np.mean(yhat != y_data)
        """
    def test(self):
        f=file_io.FileIO(os.environ["AIP_TEST_DATA_URI"], 'r')
        data=np.genfromtxt(f,delimiter=",")
        m,n=data.shape
        x_data,y_data=np.split(data,[n-1],axis=1)
        x_data_n=self.normalize(x_data)
        self.test_err = self.getErr(x_data_n,y_data)
        """
        if (self.modelConfig.model_type=="regression"):
            yhat = self.model.predict(x_data_n)
            self.test_err= mean_squared_error(y_data, yhat) / 2
        if (self.modelConfig.model_type=="classification"):
            yhat = self.model.predict(x_data_n)
            yhat = np.where(yhat >= self.threshold, 1, 0)
            self.test_err = np.mean(yhat != y_data)
        """

    def load(self):
        model_directory = os.environ['AIP_MODEL_DIR']
        self.model = tf.keras.models.load_model(model_directory)
        
    def predict(self,input):
        self.load()
        return self.model.predict(input)
        

    def normalize(self,x_data):
        norm_l = tf.keras.layers.Normalization(axis=-1)
        norm_l.adapt(x_data)  # learns mean, variance
        return norm_l(x_data)

    def getErr(self,x_data_n,y_data):
        if (self.modelConfig.model_type=="regression"):
            yhat = self.model.predict(x_data_n)
            return mean_squared_error(y_data, yhat) / 2
        if (self.modelConfig.model_type=="classification"):
            yhat = self.model.predict(x_data_n)
            yhat = np.where(yhat >= self.threshold, 1, 0)
            return np.mean(yhat != y_data)
        if (self.modelConfig.model_type=="multi-classification"):
            model_predict_lambda = lambda Xl: np.argmax(tf.nn.softmax(self.model.predict(Xl)).numpy(),axis=1)
            yhat=model_predict_lambda(x_data_n)
            m = len(y_data)
            incorrect = 0
            for i in range(m):
                if yhat[i] != y_data[i]:
                    incorrect += 1
            cerr=incorrect/m
            return(cerr)