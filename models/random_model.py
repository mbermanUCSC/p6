from models.model import Model
from tensorflow.keras import Sequential, layers, models
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.optimizers import RMSprop, Adam

import numpy as np

class RandomModel(Model):
    def _define_model(self, input_shape, categories_count):

        base_model = models.load_model('results/basic_model_10_epochs_timestamp_1708642612.keras')
        
        self._randomize_layers(base_model)
        

        self.model = Sequential(base_model.layers[:-1])  # remove the last layer
        self.model.add(layers.Flatten(name='random_flatten'))
        self.model.add(layers.Dense(128, activation='relu', name='random_dense1')) 
        self.model.add(layers.Dense(categories_count, activation='softmax', name='random_dense2'))  

    
    def _compile_model(self):
        self.model.compile(optimizer=Adam(learning_rate=0.001),
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])


    @staticmethod
    def _randomize_layers(model):
         for layer in model.layers:
            if hasattr(layer, 'weights') and layer.weights:
                w_shapes = [w.shape for w in layer.get_weights()]
                random_weights = [np.random.normal(size=shape) for shape in w_shapes]
                layer.set_weights(random_weights)
