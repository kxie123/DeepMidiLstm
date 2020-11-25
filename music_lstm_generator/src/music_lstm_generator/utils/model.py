import numpy as np
import logging
import random
from keras.models import Sequential
from keras.layers import Dense, TimeDistributed
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Activation
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras import optimizers

logging.basicConfig(level = logging.INFO)
log = logging.getLogger()


class MidiLSTM:
    def __init__(self, seq_length=60, units=512, dropout_factor=0.5, lr=.05, rho=0.9, decay=0,
                    validation_split=.05, epochs=100, batch_size=128, initial_epoch=0):
        self.seq_length = seq_length
        self.units = units
        self.dropout_factor = dropout_factor
        self.lr = lr
        self.rho = rho
        self.decay = decay
        self.validation_split = validation_split
        self.epochs = epochs
        self.batch_size = batch_size
        self.initial_epoch = initial_epoch
        self._model = None

    @property
    def model(self, dim):
        if not self._model:
            model = Sequential()
            model.add(LSTM(self.units, input_shape = (None, dim), return_sequences=True)) 
            model.add(Dropout(0.8))
            model.add(LSTM(self.units, return_sequences=True))
            model.add(Dropout(0.8))
            model.add(LSTM(self.units, return_sequences=True))
            model.add(Dropout(0.8))
            model.add(TimeDistributed(Dense(dim)))
            model.add(Activation('softmax'))
            model.compile(loss='categorical_crossentropy', optimizer = optimizers.RMSprop(lr=self.lr, rho=self.rho, decay=self.decay))
            self._model = model

        return model
    
    def build_training_sequences(self, notes, step=1):
        ''' Prepare the sequences used by the LSTM '''
    
        sequence_in = []
        sequence_out = []

        # get set of all elements in corpus
        elements = sorted(set(item for item in notes))
        # create a dictionary to map pitches to integers
        element_to_int = dict((element, number) for number, element in enumerate(elements))
        # create input sequences and the corresponding outputs
        for i in range(0, len(notes) - self.seq_length + 1, step):
            cur_seq = notes[i:i + self.seq_length]
            next_seq = notes[i+1 : i + 1 + self.seq_length]
            sequence_in.append(cur_seq)
            sequence_out.append(next_seq)

        log.info("Length of sequences: {}".format(len(sequence_in)))
        #init enconde vectors
        encoded_input = np.zeros((len(sequence_in), self.seq_length, len(elements)), dtype=np.bool)
        encoded_output = np.zeros((len(sequence_in), self.seq_length, len(elements)), dtype=np.bool)
        #encode input and output sequences
        for i, sequence in enumerate(sequence_in):
            for j, element in enumerate(sequence):
                encoded_input[i, j, element_to_int[element]] = 1
        for i, sequence in enumerate(sequence_out):
            for j, element in enumerate(sequence):
                encoded_output[i, j, element_to_int[element]] = 1

        return (len(elements), encoded_input, encoded_output)

    def train(self, model, encoded_input, encoded_output, init_epoch=0):
        #save model checkpoint after each epoch
        checkpoint = ModelCheckpoint(
            "weights_pop/epoch-{epoch:02d}-loss-{loss:.4f}-v2.hdf5",
            monitor='loss',
            verbose=0,
            save_best_only=False,
            mode='min'
        )
        callbacks_list = [checkpoint]
        model.fit(encoded_input, encoded_output, validation_split=self.validation_split, epochs=self.epochs, 
                    batch_size=self.batch_size, callbacks=callbacks_list, initial_epoch=self.epochs)

    def build_prediction_sequences(self, notes, step):
        ''' Prepare the sequences used by the Neural Network '''
    
        sequence_in = []
        sequence_out = []
        # get set of all elements in corpus
        elements = sorted(set(item for item in notes))
        # create a dictionary to map pitches to integers
        element_to_int = dict((element, number) for number, element in enumerate(elements))
        int_to_element = dict((number, element) for number, element in enumerate(elements))
        return (len(elements),element_to_int, int_to_element)

    def predict(self, model, element_to_int, int_to_element, length_prediction=500, temperature=0.3, primer_seq=None):

        if primer_seq:
            cur_elements = primer_seq
        else:
            #random primer note
            cur_elements = [str(random.randint(0,127))]
        i = 0
        while i < length_prediction:
            inputs = cur_elements
            if len(inputs)>60:
                inputs = inputs[-60:]
            x = np.zeros((1, len(inputs),model.input_shape[2]))
            for j, element in enumerate(inputs):
                x[0, j, element_to_int[element]] = 1
            predictions = model.predict(x, verbose=0)[0]
            ix = self._sample(predictions[len(inputs)-1],temperature=temperature)
            next_element = int_to_element[ix]
            cur_elements = cur_elements + [next_element] 
            i += 1

        return cur_elements

    def _sample(self, preds, temperature):
        # helper function to sample an index from a probability array (from Keras library)
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds + 1e-8) / temperature  # Taking the log should be optional? add fudge factor to avoid log(0)
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1,preds)
        #print (np.argmax(probas))
        return np.argmax(probas)
