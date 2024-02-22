from models.model import Model
from tensorflow.keras import Sequential, layers, models
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.optimizers import RMSprop, Adam

class TransferredModel(Model):
    def _define_model(self, input_shape, categories_count):

        base_model = models.load_model('results/basic_model_10_epochs_timestamp_1708642612.keras')

        # freeze the layers of the base model
        for layer in base_model.layers:
            layer.trainable = False
        

        self.model = Sequential(base_model.layers[:-1])  # remove the last layer
        

        self.model.add(layers.Flatten(name='transfer_flatten'))  
        self.model.add(layers.Dense(128, activation='relu', name='transfer_dense1')) 
        self.model.add(layers.Dense(categories_count, activation='softmax', name='transfer_dense2'))

    def _compile_model(self):
        self.model.compile(optimizer=Adam(learning_rate=0.001),
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])