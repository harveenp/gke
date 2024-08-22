import os
import numpy as np
import time
from google.cloud import storage
from tempfile import TemporaryFile

class InputDataProcessor:    
    def __init__(self,bucket_name, use_case, file_name):
        self.bucket_name = bucket_name
        self.use_case = use_case
        self.file_name = file_name
        self.storage_client = storage.Client()
        self.bucket = self.storage_client.bucket(bucket_name)
        self.raw_data_file=use_case + "/raw_data/" + file_name
        self.train_data_file=use_case + "/training_data/" + file_name
        self.dev_data_file=use_case + "/dev_data/" + file_name
        self.test_data_file=use_case + "/test_data/" + file_name
        self.orig_data_file=use_case + "/orig_data/" + file_name
    
    def prepareCsvModelInputData(self,filter_function,distribution_function):
        self.setupDirStructure()
        data=filter_function(self.bucket,self.raw_data_file)
        data_train,data_dev,data_test=distribution_function(data)

        blob_train = self.bucket.blob(self.train_data_file)
        with TemporaryFile() as temp_file_train:
            np.savetxt(temp_file_train, data_train, delimiter=",")
            temp_file_train.seek(0)
            blob_train.upload_from_file(temp_file_train)

        blob_dev = self.bucket.blob(self.dev_data_file)
        with TemporaryFile() as temp_file_dev:
            np.savetxt(temp_file_dev, data_dev, delimiter=",")
            temp_file_dev.seek(0)
            blob_dev.upload_from_file(temp_file_dev)

        blob_test = self.bucket.blob(self.test_data_file)
        with TemporaryFile() as temp_file_test:
            np.savetxt(temp_file_test, data_test, delimiter=",")
            temp_file_test.seek(0)
            blob_test.upload_from_file(temp_file_test)
        return data_train,data_dev,data_test

    def setupDirStructure(self):
        blob_raw=self.bucket.blob(self.use_case + '/raw_data/')
        blob_raw.upload_from_string('', content_type='application/x-www-form-urlencoded;charset=UTF-8')

        blob_training=self.bucket.blob(self.use_case + '/training_data/')
        blob_training.upload_from_string('', content_type='application/x-www-form-urlencoded;charset=UTF-8')

        blob_dev=self.bucket.blob(self.use_case + '/dev_data/')
        blob_dev.upload_from_string('', content_type='application/x-www-form-urlencoded;charset=UTF-8')

        blob_test=self.bucket.blob(self.use_case + '/test_data/')
        blob_test.upload_from_string('', content_type='application/x-www-form-urlencoded;charset=UTF-8')

        source_blob=self.bucket.blob(self.orig_data_file)
        self.bucket.copy_blob(source_blob,self.bucket,self.raw_data_file)
        
    def loadNumpy(self):
        print(self.orig_data_file)
        blob_raw = self.bucket.blob(self.orig_data_file)
        with blob_raw.open("rb") as f:
            data = np.load(f)
        return data
    
    def readFile(skip_header=0,delimiter=',',filling_values=0):
        file_path=use_case + "/" + file_name
        blob = bucket.blob(file_path)
        with blob_raw.open("r") as f:
            data = np.genfromtxt(f,skip_header=skip_header,delimiter=delimiter,filling_values=filling_values)
        return data