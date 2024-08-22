from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam
class ModelConfig:
    def __init__(self,model_type,layers,neurons,epochs,loss,optimizer,regularizers,verbose,no_of_networks):
        self.model_type=model_type
        self.layers=layers
        self.neurons=neurons
        self.epochs=epochs
        self.loss=loss
        self.optimizer=optimizer
        self.regularizers=regularizers
        self.verbose=verbose
        self.no_of_networks=no_of_networks
