# --------------------------------------------------------------------------
# Licensed Materials - Property of IBM
#
# (C) Copyright IBM Corp. 2019, 2020
# --------------------------------------------------------------------------

from pathlib import Path
from copy import deepcopy

import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K
import keras.layers
import keras.models
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Activation
import joblib
from sklearn.preprocessing import OneHotEncoder

class CustomModel:
    '''
    Custom model class that we use for our models. This should probably inherit from sklearn's base estimator, but it doesn't.
    
    It doesn't because we have more control this way, and aren't subject to their verbosity/complexity.

    This class contains the poor person's abstract methods - instead of using the ABCMeta and decorator, we're raising errors.

    All of these methods are required. All of the attributes put here are required. All future subclasses need these.
    '''
    # Easy reference to the config dict
    config_dict = None
    # Aliases for model names for this class
    # Easier to maintain this manually for now
    custom_aliases = {}
    # Having a reference to the experiment folder is useful
    experiment_folder = None
    def __init__(self, **kwargs):
        # References to test data
        # This is useful for tracking NN performance during training
        # and the sklearn .fit does not take two sets of data, so you cannot track test while training
        # this circumvents that if provided
        self.data_test = None
        self.labels_test = None
        self.scorer_func = None # For tracking performance
        # Required attribute for confusion matrix and other things in sklearn
        self.classes_ = None
        # Useful to have
        self.n_dims = None
        self.n_classes = None
        self.n_examples = None

    def fit(self, data, labels, save_best=True):
        '''
        sklearn style fit function

        We add an additional argument "save_best" so that we can save a model during training, and then switch saving off for plotting
        '''
        raise NotImplementedError()

    def predict(self, data):
        '''
        Function to predict labels or values
        '''
        raise NotImplementedError()

    def predict_proba(self, data):
        '''
        Function for classification to predict class probabilities
        '''
        raise NotImplementedError()

    def set_params(self, **params):
        '''
        Function used for setting parameters in both a tuning and single model setting.

        Universal so can implement here.
        '''
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"{key} is not a valid attribute of {self}")
        return self

    def get_params(self, deep=False):
        '''
        Getter function. Vastly inferior to sklearn's, but we don't really use it.

        If issues occur, then inheriting from the BaseEstimator
        '''
        return self.__dict__

    def save_model(self):
        '''
        Save the model. This may require some additional functions to deal with pickle (especially for TF).
        '''
        raise NotImplementedError()

    @classmethod
    def load_model(cls, model_path):
        '''
        Load the pickle'd model from the given path.
        '''
        raise NotImplementedError()

    @classmethod
    def setup_custom_model(cls, config_dict, experiment_folder, model_name, ref_model_dict, param_ranges, scorer_func, x_test=None, y_test=None):
        '''
        Some initial preprocessing is needed for the class to set some attributes and determine tuning type.
        '''
        cls.setup_cls_vars(config_dict, experiment_folder)
        # Add the test data if provided
        if x_test is not None:
            param_ranges["data_test"] = x_test
        if y_test is not None:
            param_ranges["labels_test"] = y_test

        param_ranges["scorer_func"] = scorer_func
        # Determine whether we can use hyper_tuning or not
        try:
            ref_model_dict[model_name]
            single_model_flag = False
        except KeyError:
            single_model_flag = True
            print(f"No parameter definition for {model_name} using {config_dict['hyper_tuning']}, using single model instead")
        return single_model_flag, param_ranges

    @classmethod
    def setup_cls_vars(cls, config_dict, experiment_folder):
        # Refer to the config_dict in the class
        cls.config_dict = config_dict
        # Give access to the experiment_folder
        cls.experiment_folder = experiment_folder

    def __repr__(self):
        return f"{self.__class__.__name__} model with params:{ {k:v for k,v in self.__dict__.items() if 'data' not in k if 'label' not in k} }"

class MLPEnsemble(CustomModel):
    #
    nickname = "mlp_ens"
    # Attributes from the config
    config_dict = None
    def __init__(self, **kwargs):
        # Param attributes
        self.n_estimators = None
        self.n_epochs = None
        self.batch_size = None
        self.lr = None # Learning rate
        self.layer_sizes = None
        self.verbose = None
        self.random_state = None
        self.scorer_func = None
        # Attributes for the ensemble
        self.data = None
        self.data_test = None # To track performance over epochs
        self.labels = None
        self.labels_test = None
        self.n_classes = None
        self.n_examples = None
        self.n_dims = None
        self.onehot_encode_obj = None
        self.classes_ = None
        # TF attributes
        self.sess = None
        self.graph = None
        self.saver = None
        # TF containers
        self.out_layers = None
        self.predictions = None
        self.costs = None
        self.optimizers = None
        self.accs = None


    def fit(self, data, labels, save_best=True):
        '''
        Fit to provided data and labels
        '''
        # Setup some of the attributes from the data
        self.data = data
        self.labels = labels
        self.n_examples = data.shape[0]
        self.n_dims = data.shape[1]
        # Set up the needed things for training now we have access to the data and labels
        self._preparation()
        # Determine the number of classes
        if self.config_dict["problem_type"] == "classification":
            # One-hot encoding has already been done, so take the info from there
            self.n_classes = self.labels.shape[1]
        elif self.config_dict["problem_type"] == "regression":
            self.n_classes = 1
        # Can add warm start logic here in the future if needed
        # Set the seeds
        np.random.seed(self.random_state)
        tf.random.set_random_seed(self.random_state)
        # Get the graph object for easy access
        self.graph = tf.Graph()
        with self.graph.as_default():
            # Setup the graph
            init = self._define_graph()
            # Create and assign the session
            self.sess = tf.Session(graph=self.graph)
            # Actually run the initializer
            self.sess.run(init)
            # Train the model
            self._train(labels, save_best)
        return self
        

    def _define_graph(self):
        # Create containers (we can't pickle them)
        self.out_layers = {}
        self.predictions = {}
        self.costs = {}
        self.optimizers = {}
        self.accs = {}
        # Define data and label placeholders
        data_pl = tf.placeholder(tf.float32, shape=(None, self.n_dims), name="data")
        labels_pl = tf.placeholder(tf.float32, shape=(None, self.n_classes), name="labels")
        # Setup the weight initialization (using He initialization)
        he_init = tf.contrib.layers.variance_scaling_initializer(
            factor=2.0,
            mode="FAN_IN",
            uniform=False
        )
        if self.config_dict["problem_type"] == "classification":
            # Matrix for ensemble prediction (geometric mean)
            ens_pred = tf.ones(shape=[tf.shape(labels_pl)[0], self.n_classes])
        elif self.config_dict["problem_type"] == "regression":
            ens_pred = tf.zeros(shape=[tf.shape(labels_pl)[0], self.n_classes])
        # Placeholder for batchnorm training flag
        # Default it to True, so then we just can pass False during the .predict
        training_pl = tf.placeholder_with_default(True, shape=(), name="is_training")
        # Setup the variables for each model
        for m in range(int(self.n_estimators)):
            # Set reuse to be True so we can use get_variable to retrieve varaibles too
            with tf.variable_scope(f"model_{m}", reuse=tf.AUTO_REUSE):
                # Loop over the given layer sizes
                for i, layer_size in enumerate(self.layer_sizes):
                    # Specify initial layer as a function of the input
                    if i == 0:
                        # Initialize hidden layer
                        tf.get_variable(
                            f"hidden_{i}", shape=(self.n_dims, layer_size),
                            dtype=tf.float32,
                            initializer=he_init
                        )
                        # Specify the layer calc
                        hidden_layer = tf.nn.relu(
                            tf.matmul(data_pl, tf.get_variable(f"hidden_{i}"))
                        )
                        # Add some batchnorm
                        tf.layers.batch_normalization(
                            hidden_layer,
                            name=f"batchborm_{i}",
                            training=training_pl
                        )                        
                    # Otherwise it's relative the the layer before it
                    else:
                        # Initialize hidden layer
                        tf.get_variable(
                            f"hidden_{i}", shape=(self.layer_sizes[i-1], layer_size),
                            dtype=tf.float32,
                            initializer=he_init
                        )
                        # Specify the layer calc
                        hidden_layer = tf.nn.relu(
                            tf.matmul(hidden_layer, tf.get_variable(f"hidden_{i}"))
                        )
                        # Add some batchnorm
                        tf.layers.batch_normalization(
                            hidden_layer,
                            name=f"batchborm_{i}",
                            training=training_pl
                        )
                # Create the out (final) layer
                tf.get_variable(
                    "out", shape=(self.layer_sizes[-1], self.n_classes),
                    dtype=tf.float32,
                    initializer=he_init
                )                
                # Combine weights for the final layer
                self.out_layers[m] = tf.matmul(
                    hidden_layer, tf.get_variable("out")
                )
                # Select the appropriate combination and loss function
                if self.config_dict["problem_type"] == "classification":
                    # Use softmax to combine the models
                    self.predictions[m] = tf.nn.softmax(self.out_layers[m])
                    # Cross entropy loss
                    self.costs[m] = tf.nn.softmax_cross_entropy_with_logits_v2(
                        labels=labels_pl, logits=self.out_layers[m])
                elif self.config_dict["problem_type"] == "regression":
                    # No combination method needed for regression
                    self.predictions[m] = self.out_layers[m]
                    # MSE loss function
                    self.costs[m] = tf.losses.mean_squared_error(
                        labels=labels_pl, predictions=self.predictions[m])
                # Define the optimizer
                self.optimizers[m] = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.costs[m])
                # # Container for the accuracies
                # self.accs[m] = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.predictions[m], 1), tf.argmax(labels_pl, 1)), "float"))
            # Combine the labels
            if self.config_dict["problem_type"] == "classification":
                # Calculate the geometric mean (iteratively)
                ens_pred = tf.multiply(ens_pred, tf.pow(self.predictions[m], 1/self.n_estimators))
            elif self.config_dict["problem_type"] == "regression":
                ens_pred = tf.add(ens_pred, self.predictions[m])
        # Normalize the ensemble prediction (for the normalized geometric mean)
        if self.config_dict["problem_type"] == "classification":
            ensemble = tf.divide(ens_pred, tf.reduce_sum(ens_pred, 1, keepdims=True), name="ensemble")
        # Divide through to get the arithmetic mean
        elif self.config_dict["problem_type"] == "regression":
            ensemble = tf.divide(ens_pred, float(self.n_estimators), name="ensemble")
        # Initialize the variables
        init = tf.global_variables_initializer()
        return init


    def _train(self, labels, save_best):
        # Get some tensors from the graph
        data_pl = self.graph.get_tensor_by_name("data:0")
        labels_pl = self.graph.get_tensor_by_name("labels:0")
        ensemble = self.graph.get_tensor_by_name("ensemble:0")
        training_pl = self.graph.get_tensor_by_name("is_training:0")
        # Create saver object
        self.saver = tf.train.Saver(max_to_keep=1, save_relative_paths=True)
        # Greater is better is the convention, so get the lowest number possible
        best_score = -np.inf
        # Loop over our epochs
        for epoch in range(int(self.n_epochs)):
            # Loop over our batches
            for ex_num in range(0, self.n_examples, self.batch_size):
                # Each model is run sequentially
                for m in range(int(self.n_estimators)):
                    # Setup the batch examples and labels
                    example_batch = self.data[ex_num:ex_num+self.batch_size, :]
                    label_batch = self.labels[ex_num:ex_num+self.batch_size]
                    # Train the model
                    _, cost = self.sess.run([self.optimizers[m], self.costs[m]], feed_dict={data_pl: example_batch, labels_pl: label_batch})
            if save_best:
                # Access the attributes of the scorer_func so that we can call the underlying scoring function
                # This avoids having to pass an estimator into the scorer
                score = self.scorer_func._score_func(
                    labels,
                    self.predict(self.data),
                    **self.scorer_func._kwargs # Extract the kwargs and pass them in
                ) * self.scorer_func._sign # Multiply by the sign so that greater is better is maintained
                # Save the best score
                if score > best_score:
                    print(f"Found new best score ({self.scorer_func._score_func.__name__}) with {np.abs(score)}")
                    best_score = score
                    self.save_model()
            # Print progress
            if self.verbose:
                print(f"Epoch: {epoch}")

        print("Training finished!")


    def predict(self, data):
        '''
        Class label prediction (similar to scikit-learn), or regression value prediction
        '''
        ensemble = self.graph.get_tensor_by_name("ensemble:0")
        data_pl = self.graph.get_tensor_by_name("data:0")
        labels_pl = self.graph.get_tensor_by_name("labels:0")
        training_pl = self.graph.get_tensor_by_name("is_training:0")
        if self.config_dict["problem_type"] == "classification":
            # 
            pred_vals = np.argmax(ensemble.eval({data_pl:data, labels_pl:np.zeros((data.shape[0], self.n_classes)), training_pl:False}, session=self.sess), 1)
            preds = self.onehot_encode_obj.categories_[0][pred_vals]
        elif self.config_dict["problem_type"] == "regression":
            preds = ensemble.eval({data_pl:data, labels_pl:np.zeros((data.shape[0], self.n_classes)), training_pl:False}, session=self.sess)
        return preds.flatten()
    

    def predict_proba(self, data):
        '''
        Class probability prediction
        '''
        if self.config_dict["problem_type"] == "classification":
            # Get some tensors from the graph
            ensemble = self.graph.get_tensor_by_name("ensemble:0")
            data_pl = self.graph.get_tensor_by_name("data:0")
            labels_pl = self.graph.get_tensor_by_name("labels:0")
            training_pl = self.graph.get_tensor_by_name("is_training:0")
            # We need to provide some labels for the pipeline, but they are not used
            # So just provide an array of zeros
            return ensemble.eval({data_pl:data, labels_pl:np.zeros((data.shape[0], self.n_classes)), training_pl:False}, session=self.sess)
        else:
            raise NotImplementedError()
    
    
    def set_params(self, **params):
        '''
        Required function (in sklearn BaseEstimator) used for setting parameters in both a tuning and single model setting
        '''
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"{key} is not a valid attribute of {self}")
        return self
    

    def get_params(self, deep=False):
        '''
        For consistency with sklearn - this may need be expanded a bit for full compatability
        '''
        return self.__dict__


    def save_model(self):
        # Construct the file name
        fname = f"{self.experiment_folder / 'models' / 'mlp_ens_best'}"
        # Save the session
        self.saver.save(self.sess, fname)
        # Pickle the instance
        self._pickle_member(fname)


    def _pickle_member(self, fname):
        '''
        Custom function to pickle an instance. 
        '''
        # Remove the TF params
        removed_params = self._make_picklable()
        # Pickle the now TF-free object
        with open(fname+".pkl", 'wb') as f:
            joblib.dump(self, f)
        # Restore the TF params
        self._restore_instance(removed_params)


    def _restore_instance(self, removed_params):
        # Restore attributes
        for attr, value in removed_params.items():
            setattr(self, attr, value)


    def __copy__(self):
        new = type(self)()
        new.__dict__.update(self.__dict__)
        return new


    def __deepcopy__(self, memo):
        '''
        Implement deepcopy dunder method as we can't deepcopy tensorflow objects
        '''
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        removed_params = self._remove_tf_params

        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        
        self._make_picklable(removed_params)
        result._make_picklable(removed_params)
        return result


    def _make_picklable(self):
        # Create a temp container
        removed_params = {}
        # Loop over the TF attributes to set to None for pickle
        for attr in [
            "sess", "graph", "out_layers", "predictions",
            "costs", "optimizers", "accs", "saver",
            "data", "labels", "data_test", "labels_test"]:
            removed_params[attr] = getattr(self, attr)
            setattr(self, attr, None)
        return removed_params


    @classmethod
    def load_model(cls, model_path):
        '''
        Load a previously saved model
        '''
        model_path = str(model_path)
        # Load the pickled instance
        with open(model_path+".pkl", 'rb') as f:
            model = joblib.load(f)
        # Reset the default graph just in case
        tf.reset_default_graph()
        # Load the TF graph and session
        model.saver = tf.train.import_meta_graph(model_path+".meta")
        # model.graph = tf.Graph()
        model.graph = tf.get_default_graph()
        # model.graph.clear_collection("local_variables")
        with model.graph.as_default():
            # Define the session
            model.sess = tf.Session(graph=model.graph)
            model.sess.run(tf.global_variables_initializer())
            model.saver.restore(model.sess, tf.train.latest_checkpoint(Path(model_path).parents[0]))
        return model


    def _onehot_encode(self):
        '''
        Our network requires one-hot encoded class labels, so do that if it isn't done already
        '''
        # Check if labels are not already encoded (somehow)
        # And check whether we have done any encoding yet
        if len(self.labels.shape) == 1 or self.labels.shape[1] > 1:
            # Create the encode object
            self.onehot_encode_obj = OneHotEncoder(categories='auto', sparse=False)
            # Fit transform the labels that we have
            # Reshape the labels just in case (if they are, it has no effect)
            self.labels = self.onehot_encode_obj.fit_transform(self.labels.reshape(-1,1))
            # Set the classes for the model (useful for the plotting e.g. confusion matrix)
            self.classes_ = self.onehot_encode_obj.categories_[0]
            # Transform the test labels if we have them
            if self.labels_test is not None:
                self.labels_test = self.onehot_encode_obj.transform(self.labels_test.reshape(-1,1))


    def _preparation(self):
        # Convert the labels if a DataFrame/Series
        if isinstance(self.labels, (pd.DataFrame, pd.Series)):
            self.labels = self.labels.values
        # Same for the test labels
        if self.labels_test is not None:
            if isinstance(self.labels_test, (pd.DataFrame, pd.Series)):
                self.labels_test = self.labels_test.values
        # Check if we need to one-hot encode
        if self.config_dict["problem_type"] == "classification":
            self._onehot_encode()
        elif self.config_dict["problem_type"] == "regression":
            self.labels = self.labels.reshape(-1, 1)
            if self.labels_test is not None:
                self.labels_test = self.labels_test.reshape(-1, 1)


class MLPKeras(CustomModel):
    nickname = "mlp_keras"
    # Attributes from the config
    config_dict = None
    def __init__(self, n_epochs=None, batch_size=None, lr=None, layer_dict=None,
                verbose=None, random_state=None, n_blocks=None, dropout=None,
                scorer_func=None, data=None, data_test=None, labels=None, labels_test=None,
                n_classes=None, n_examples=None, n_dims=None, onehot_encode_obj=None,
                classes_=None, model=None):
        # Param attributes
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr # Learning rate
        self.layer_dict = layer_dict
        self.verbose = verbose
        self.random_state = random_state
        self.n_blocks = n_blocks
        self.dropout = dropout
        self.scorer_func = scorer_func
        # Attributes for the model
        self.data = data
        self.data_test = data_test # To track performance over epochs
        self.labels = labels
        self.labels_test = labels_test
        self.n_classes = n_classes
        self.n_examples = n_examples
        self.n_dims = n_dims
        self.onehot_encode_obj = onehot_encode_obj
        self.classes_ = classes_
        # Keras attributes
        self.model = model


    def fit(self, data, labels, save_best=True):
        '''
        Fit to provided data and labels
        '''
        # Setup some of the attributes from the data
        self.data = data
        self.labels = labels
        self.n_examples = data.shape[0]
        self.n_dims = data.shape[1]
        # Set up the needed things for training now we have access to the data and labels
        self._preparation()
        # Determine the number of classes
        if self.config_dict["problem_type"] == "classification":
            # One-hot encoding has already been done, so take the info from there
            self.n_classes = self.labels.shape[1]
        elif self.config_dict["problem_type"] == "regression":
            self.n_classes = 1
        # Define the model
        self._define_model()
        # Define the optimizer
        '''This is Cameron's optimizer
            adam_opt = keras.optimizers.Adam(
            lr=self.lr,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=None,
            decay=0.0,
            amsgrad=False
        )'''
        #Sean's optimizer 
        rmsprop = keras.optimizers.rmsprop(lr=self.lr, rho=0.9, epsilon=0.1, decay=0.000001) 
        # Compile the model
        if self.config_dict["problem_type"] == "classification":
            self.model.compile(
                optimizer=rmsprop,
                loss='categorical_crossentropy'
            )
        elif self.config_dict["problem_type"] == "regression":
            self.model.compile(
                optimizer=rmsprop,
                loss='mae'
            )
        # Set the verbosity level
        if self.verbose:
            verbose = 1
        else:
            verbose = 0
        best_score = -np.inf
        # Loop over the epochs so we can do some inspection after each
        for i in range(int(self.n_epochs)):
            if self.verbose:
                print(f"Epoch: {i}")
            # Fit the model
            self.model.fit(self.data, self.labels, epochs=1, batch_size=self.batch_size, verbose=verbose)
            
            # # If test data is provided, see how the performance is
            # if self.data_test is not None and self.labels_test is not None:
            #     # print("Evaluating on test data")
            #     score = self.model.evaluate(self.data_test, self.labels_test, batch_size=self.batch_size, verbose=verbose)
            
            if save_best:
                # Access the attributes of the scorer_func so that we can call the underlying scoring function
                # This avoids having to pass an estimator into the scorer
                score = self.scorer_func._score_func(
                    labels,
                    self.predict(self.data),
                    **self.scorer_func._kwargs # Extract the kwargs and pass them in
                ) * self.scorer_func._sign # Multiply by the sign so that greater is better is maintained
                # Save the best score
                if score > best_score:
                    print(f"Found new best score ({self.scorer_func._score_func.__name__}) with {np.abs(score)}")
                    best_score = score
                    self.save_model()
        if self.verbose:
            print("Model Summary:")
            print(self.model.summary())
        return self
    

    def _define_model(self):
        '''
        Architecture created by Sean
        '''
        model = Sequential()
        model.add(Dense(self.n_dims, input_dim=self.n_dims, kernel_initializer='uniform'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        # Add multiple blocks of this format
        for i in range(int(self.n_blocks)):
            model.add(BatchNormalization())
            model.add(Dense(100, kernel_initializer='uniform'))
            model.add(Activation('relu'))
        # Add some dropout at the end
        model.add(Dropout(self.dropout))
        # Define the output layer differently for the problem type
        if self.config_dict["problem_type"] == "classification":
            model.add(Dense(
                self.n_classes,
                kernel_initializer=keras.initializers.glorot_normal(seed=self.config_dict["seed_num"])
            ))
            model.add(Activation(tf.nn.softmax)) # The tensorflow softmax?
        elif self.config_dict["problem_type"] == "regression":
            model.add(Dense(
                self.n_classes,
                kernel_initializer=keras.initializers.glorot_normal(seed=self.config_dict["seed_num"]),
                activation='linear'
            ))
        # Assign the model
        self.model = model


    def predict(self, data):
        if self.config_dict["problem_type"] == "classification":
            pred_inds = np.argmax(self.model.predict(data), axis=1)
            preds = self.onehot_encode_obj.categories_[0][pred_inds]
        elif self.config_dict["problem_type"] == "regression":
            preds = self.model.predict(data)
        return preds.flatten()


    def predict_proba(self, data):
        if self.config_dict["problem_type"] == "classification":
            return self.model.predict(data)
        else:
            raise NotImplementedError()


    def set_params(self, **params):
        '''
        Required function (in sklearn BaseEstimator) used for setting parameters in both a tuning and single model setting
        '''
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"{key} is not a valid attribute of {self}")
        return self
    

    def save_model(self):
        fname = f"{self.experiment_folder / 'models' / 'mlp_keras_best'}"
        self.model.save(fname+".model")
        self._pickle_member(fname)
    

    def _pickle_member(self, fname):
        '''
        Custom function to pickle an instance. 
        '''
        # Create a temp container
        temp_params = {}
        # Loop over the TF attributes to set to None for pickle
        for attr in ["model"]:
            temp_params[attr] = getattr(self, attr)
            setattr(self, attr, None)
        # Pickle the now TF-free object
        with open(fname+".pkl", 'wb') as f:
            joblib.dump(self, f)
        # Restore attributes
        for attr, value in temp_params.items():
            setattr(self, attr, value)


    @classmethod
    def load_model(cls, model_path):
        model_path = str(model_path)
        # Load the pickled instance
        with open(model_path+".pkl", 'rb') as f:
            model = joblib.load(f)
        # Load the model with Keras and set this to the relevant attribute
        model.model = keras.models.load_model(model_path+".model")
        return model


    def _onehot_encode(self):
        '''
        Our network requires one-hot encoded class labels, so do that if it isn't done already
        '''
        # Check if the labels are already one-hot encoded
        if len(self.labels.shape) == 1 or self.labels.shape[1] > 1:
            # Create the encode object
            self.onehot_encode_obj = OneHotEncoder(categories='auto', sparse=False)
            # Fit transform the labels that we have
            # Reshape the labels just in case (if they are, it has no effect)
            self.labels = self.onehot_encode_obj.fit_transform(self.labels.reshape(-1,1))
            # Set the classes for the model (useful for the plotting e.g. confusion matrix)
            self.classes_ = self.onehot_encode_obj.categories_[0]
            # Transform the test labels if we have them
            if self.labels_test is not None:
                self.labels_test = self.onehot_encode_obj.transform(self.labels_test.reshape(-1,1))


    def _preparation(self):
        # Convert the labels if a DataFrame/Series
        if isinstance(self.labels, (pd.DataFrame, pd.Series)):
            self.labels = self.labels.values
        # Same for the test labels
        if self.labels_test is not None:
            if isinstance(self.labels_test, (pd.DataFrame, pd.Series)):
                self.labels_test = self.labels_test.values
        # Check if we need to one-hot encode
        if self.config_dict["problem_type"] == "classification":
            self._onehot_encode()
        elif self.config_dict["problem_type"] == "regression":
            self.labels = self.labels.reshape(-1, 1)
            if self.labels_test is not None:
                self.labels_test = self.labels_test.reshape(-1, 1)
