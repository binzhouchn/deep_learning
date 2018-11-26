# coding=utf-8

from keras import Input, Model
from keras.layers import Embedding, Dense, Conv1D, GlobalMaxPooling1D, Concatenate, Dropout


class TextCNN(object):
    def __init__(self, maxlen, max_features, embedding_dims,
                 weights=None,
                 trainable=True,
                 class_num=1,
                 last_activation='sigmoid',
                 optimizer='adam',
                 loss='binary_crossentropy',
                 metrics=None,
                 batch_size=64,
                 epochs=30,
                 callbacks=None):
        self.maxlen = maxlen
        self.max_features = max_features
        self.embedding_dims = embedding_dims
        self.weights = weights  # weight list
        self.trainable = trainable
        self.class_num = class_num
        self.last_activation = last_activation
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics  # metric list, calculated at end of a batch, always useless
        self.batch_size = batch_size
        self.epochs = epochs
        self.callbacks = callbacks  # callbacks list, e.g., model checkpoint, early stopping, customed epoch-level metric...

    def fit(self, x_train, x_test, y_train, y_test):
        input = Input((self.maxlen,))
        embedding = Embedding(self.max_features, self.embedding_dims,
                              input_length=self.maxlen,
                              weights=self.weights,
                              trainable=self.trainable)(input)
        convs = []
        for kernel_size in [3, 4, 5]:
            c = Conv1D(128, kernel_size, activation='relu')(embedding)
            c = GlobalMaxPooling1D()(c)
            convs.append(c)
        x = Concatenate()(convs)
        x = Dropout(0.5)(x)
        output = Dense(self.class_num, activation=self.last_activation)(x)
        self.model = Model(inputs=input, outputs=output)

        self.model.compile(self.optimizer, loss=self.loss, metrics=self.metrics)

        self.model.fit(x_train, y_train,
                       validation_data=(x_test, y_test),
                       batch_size=self.batch_size,
                       epochs=self.epochs,
                       callbacks=self.callbacks)

        return self.model

    def predict(self, x):
        return self.model.predict(x)