import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Flatten, Conv1D
from tensorflow.keras import optimizers, losses
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint, TerminateOnNaN

import autokeras as ak
from .base_model import BaseModel
from .batch_generator_seq_array import BatchGeneratorSeqArray


def to_matrix(data, n):
    return [data[i:i+n] for i in range(0, len(data), n)]


class KerasModel(BaseModel):

    def __init__(self, input_dim, output_dim, dataset_type, method='train_dnn_keras', init_model=None, conv1d=False,
                 config=None, random_state=1234):
        super().__init__(input_dim, output_dim, dataset_type)
        self.method = method
        self.init_model = init_model
        self.conv1d = conv1d
        self.config = config if config else {}
        self.random_state = random_state
        if self.method == 'train_dnn_keras':
            self.__init_fx__(input_dim, output_dim, dataset_type)
        elif self.method == 'train_dnn_autokeras':
            self.__init_ak__(input_dim, output_dim, dataset_type)

    def __init_ak__(self, input_dim, output_dim, dataset_type):
        self.epochs = self.config.get("n_epochs", 200)
        self.batch_size = self.config.get("batch_size", 32)
        self.lr = self.config.get("lr", 0.001)
        self.verbose = self.config.get("verbose", False)
        self.num_layers = self.config.get("n_blocks", 4)
        self.dropout = self.config.get("dropout", 0.3)
        self.use_batchnorm = self.config.get("use_batchnorm", True)
        self.max_trials = self.config.get("n_trials", 20)
        self.tuner = self.config.get("tuner", "greedy")

        print("self.config=", self.config)

        tmp_path = "/tmp/autokeras_{}".format(os.getpid())
        os.system("rm -rf /tmp/autokeras_{}".format(os.getpid()))

        if self.dataset_type == "regression":
            input_node = ak.Input()
            output_node = input_node
            output_node = ak.DenseBlock(num_layers=self.num_layers, dropout=self.dropout, use_batchnorm=self.use_batchnorm)(output_node)
            output_node = ak.RegressionHead(dropout=self.dropout, metrics=['mae'])(output_node)
            model = ak.AutoModel(inputs=input_node, outputs=output_node, directory=tmp_path, max_trials=self.max_trials,
                                 objective="val_loss", tuner=self.tuner, seed=self.random_state)

            """
            # The following code is not supported well yet by Autokeras :(
            metrics = ["mean_absolute_error"]  # "mean_squared_error"]
            model = ak.StructuredDataRegressor(output_dim=output_dim,
                                               directory=tmp_path,
                                               loss="mean_absolute_error",
                                               max_trials=self.max_trials,
                                               metrics = metrics,
                                               objective = "val_loss",
                                               tuner = self.tuner)
            """
        else:  # "classification"
            input_node = ak.Input()
            output_node = input_node
            output_node = ak.DenseBlock(num_layers=self.num_layers, dropout=self.dropout, use_batchnorm=self.use_batchnorm)(output_node)
            output_node = ak.ClassificationHead(multi_label=True, dropout=self.dropout, metrics=['accuracy'])(output_node)
            model = ak.AutoModel(inputs=input_node, outputs=output_node, directory=tmp_path, max_trials=self.max_trials,
                                 objective="accuracy", tuner=self.tuner, seed=self.random_state)

        self.model = model

    def __init_fx__(self, input_dim, output_dim, dataset_type):
        # import os
        # os.environ['PYTHONHASHSEED']=str(self.random_state)
        # os.environ['TF_CUDNN_DETERMINISTIC'] = str(self.random_state)
        from tensorflow.random import set_seed
        set_seed(self.random_state)
        from numpy.random import seed
        seed(self.random_state)
        import random
        random.seed(self.random_state)
        
        if self.init_model is not None:
            # init_model = self.init_model
            # init_model.summary()
            # config = init_model.get_config()
            # model = keras.Model.from_config(config)
            model = self.init_model
        else:
            # create model
            model = Sequential()

            # choose weight initializer (not important)
            # initializer = 'he_uniform'
            initializer = 'normal'

            # choose neural network architecture: number of layers, neurons per layer, activation function
            if self.dataset_type == "regression":
                if output_dim > 10:
                    model.add(Dense(256, input_dim=input_dim, kernel_initializer=initializer, activation='relu'))
                    model.add(Dense(256, kernel_initializer=initializer, activation='relu'))
                    model.add(Dense(256, kernel_initializer=initializer, activation='relu'))
                    model.add(Dense(256, kernel_initializer=initializer, activation='relu'))
                    model.add(Dense(output_dim, kernel_initializer=initializer, activation="linear"))
                else:
                    model.add(Dense(256, input_dim=input_dim, kernel_initializer=initializer, activation='relu'))
                    model.add(Dense(128, kernel_initializer=initializer, activation='relu'))
                    model.add(Dense(64, kernel_initializer=initializer, activation='relu'))
                    model.add(Dense(32, kernel_initializer=initializer, activation='relu'))
                    model.add(Dense(output_dim, kernel_initializer=initializer, activation="linear"))
            else:
                if not self.conv1d:
                    model.add(Dense(256, input_dim=input_dim, kernel_initializer=initializer, activation='relu'))
                    model.add(Dense(128, kernel_initializer=initializer, activation='relu'))
                    model.add(Dense(32, kernel_initializer=initializer, activation='relu'))
                    model.add(Dense(output_dim, kernel_initializer=initializer, activation="softmax"))
                else:
                    model.add(Conv1D(filters=16, kernel_size=1, activation='relu', input_shape=(input_dim, 1)))
                    model.add(Conv1D(filters=8, kernel_size=1, activation='relu'))
                    model.add(Flatten())
                    model.add(Dense(64, kernel_initializer=initializer, activation='relu'))
                    model.add(Dense(32, kernel_initializer=initializer, activation='relu'))
                    model.add(Dense(output_dim, kernel_initializer=initializer, activation="softmax"))

        # choose optimizer
        opt = optimizers.Adam()

        # choose loss function
        if self.dataset_type == "regression":
            # loss = losses.logcosh  # napoli
            # loss = losses.mean_squared_error
            loss = losses.mean_absolute_error  # torino, company, uom
            # loss = losses.mean_absolute_percentage_error
        else:
            loss = 'categorical_crossentropy'

        # Compile the model
        metrics = []
        if self.dataset_type == "classification":
            metrics.append('accuracy')

        model.compile(loss=loss, optimizer=opt, metrics=metrics)
        model.summary()
        self.model = model

    @staticmethod
    def _lr_schedule(epoch):
        # CHOICE 1: learning rate schedule
        lr = 1.0e-3
        factor = 1
        if epoch > 180*factor:
            lr *= 0.5e-3
        elif epoch > 160*factor:
            lr *= 1e-3
        elif epoch > 120*factor:
            lr *= 1e-2
        elif epoch > 80*factor:
            lr *= 1e-1
        print('Learning rate: ', lr)
        return lr

    def fit_data_ak(self, trainX, trainY, testX, testY, input_list=None):
        print("training AutoKeras model...")
        lr_scheduler = LearningRateScheduler(self._lr_schedule)
        callbacks = [lr_scheduler]
        callbacks.append(TerminateOnNaN())

        # train the model
        # choose number of epochs and batch_size
        epochs = self.epochs
        batch_size = self.batch_size

        self.model.fit(x=trainX, y=trainY, batch_size=batch_size, epochs=epochs,
                       validation_data=(testX, testY), callbacks=callbacks)
        exported_model = self.model.export_model()
        print(exported_model)
        exported_model.summary()
        print("Evaluating...")
        self.model.evaluate(testX, y=testY)
        print("Evaluated...")

        del self.model
        self.model = exported_model

    def fit_data_fx(self, trainX, trainY, testX, testY, input_list=None):
        lr_scheduler = LearningRateScheduler(self._lr_schedule)
        es = EarlyStopping(monitor='val_loss', verbose=1, patience=20)

        # ckpt = ModelCheckpoint('keras_model_ckpt.h5', monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto')
        ckpt = ModelCheckpoint('keras_model_ckpt_{}.h5'.format(os.getpid()), monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
        callbacks = [lr_scheduler, ckpt, es]
        # train the model
        # choose number of epochs and batch_size
        epochs = 50*2
        batch_size = 32

        print("training Keras model...")
        # history = self.model.fit(trainX, trainY, validation_data=(testX, testY),
        #                         epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=2)

        bg_train = BatchGeneratorSeqArray(trainX, trainY, batch_size=batch_size)
        if ((testX is None) and (testY is None)):
            bg_val = None
        else:
            bg_val = BatchGeneratorSeqArray(testX, testY, batch_size=batch_size)
        history = self.model.fit_generator(generator=bg_train, validation_data=bg_val,
                                           epochs=epochs, callbacks=callbacks, verbose=2, workers=1)

        # Plot training & validation loss values
        plt.plot(history.history['loss'])
        if bg_val is not None:
            plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        if bg_val is not None:
            plt.legend(['Train', 'Test'], loc='upper right')
        else:
            plt.legend(['Train'], loc='upper right')
        # plt.show()
        plt.savefig('history_{}.png'.format(os.getpid()))

        # try:
        #     ohe = self.model_ohe
        #     del self.model
        #     self.model = load_model('keras_model_ckpt_{}.h5'.format(os.getpid()))
        #     self.model_ohe = ohe
        # except BaseException as e:
        #     print("exception: ", str(e))
        #     del self.model
        #     self.model = load_model('keras_model_ckpt_{}.h5'.format(os.getpid()))

        feature_importances = None
        self.model.feature_importances_ = feature_importances

    def fit_data(self, trainX, trainY, testX=None, testY=None, input_list=None):
        if self.method == 'train_dnn_keras':
            return self.fit_data_fx(trainX, trainY, testX, testY, input_list)
        elif self.method == 'train_dnn_autokeras':
            return self.fit_data_ak(trainX, trainY, testX, testY, input_list)

    def predict(self, x):
        print("predicting values ...")
        if self.conv1d:
            x = x.reshape((x.shape[0], x.shape[1], 1))

        if self.dataset_type == "classification":
            y = self.model.predict(x)
            # y_pred = np.argmax(y, axis=-1)
            y_pred = y
        else:
            y_pred = self.model.predict(x)

        if self.output_dim == 1:
            y_pred = y_pred.reshape(-1, 1)

        return y_pred

    def predict_proba(self, x):
        print("predicting probs ...")
        if self.conv1d:
            x = x.reshape((x.shape[0], x.shape[1], 1))

        y_pred = self.model.predict(x)

        if self.output_dim == 1:
            y_pred = y_pred.reshape(-1, 1)

        return y_pred

    def save(self, path):
        if path:
            self.model.save('{}'.format(path))

    def summary(self):
        self.model.summary()
