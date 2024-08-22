import numpy as np
from google.cloud import storage

def csv_weather_data_prediction_filter(bucket,raw_data_file):
    blob_raw = bucket.blob(raw_data_file)
    with blob_raw.open("r") as f:
        data = np.genfromtxt(f,skip_header=2,delimiter=",",filling_values=0,usecols=(1,2,3,6,7,9,4))
    return data
        
    
def csv_weather_data_classification_filter(bucket,raw_data_file):
    blob_raw = bucket.blob(raw_data_file)
    with blob_raw.open("r") as f:
        data = np.genfromtxt(f,skip_header=2,delimiter=",",filling_values=0,usecols=(1,2,3,6,7,9,4))
    data[:,6]=np.where(data[:,6] > 2,1.0,0.0)
    return data

def coffee_filter(bucket,raw_data_file):
    blob_raw = bucket.blob(raw_data_file)
    with blob_raw.open("r") as f:
        data = np.genfromtxt(f,skip_header=0,delimiter=",",filling_values=0,usecols=(0,1,2))
    return data

def generic_filter(bucket_name,file_path,usecols,skip_header,delimiter):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob_raw = bucket.blob(file_path)
    with blob_raw.open("r") as f:
        if(skip_header==0):
            data = np.genfromtxt(f,delimiter=delimiter,usecols=usecols,dtype=None,encoding='ASCII')
        else:
            data = np.genfromtxt(f,skip_header=skip_header,delimiter=delimiter,filling_values=0,usecols=usecols)
    return data