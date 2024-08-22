import numpy as np
def distribute_602020(data):
    m,n=data.shape
    return np.split(data,[int(m*0.6),int(m*0.8)],axis=0)

def distribute_100(data):
    m,n=data.shape
    return np.split(data,[int(m*1),int(m*1)],axis=0)