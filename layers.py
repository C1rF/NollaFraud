import tensorflow as tf
from tensorflow import keras
from keras.layers import Layer


class IntraAgg(Layer):
    
    def __init__(self, input_dim):
        
        super().__init__(trainable=False)
        
        self.input_dim = input_dim
        
    def call(self, inputs, *args, **kwargs):
        pass