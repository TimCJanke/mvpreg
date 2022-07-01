"""
@author: Tim Janke, Energy Information Networks & Systems @ TU Darmstadt
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, StandardScaler, QuantileTransformer


from .copulas import GaussianCopula, IndependenceCopula, SchaakeShuffle, VineCopula
from .helpers import rank_data_random_tiebreaker


################ Base class for all MVPReg models ################
class MVPRegModel(object):
    
    def __init__(self,
                 dim_in=None,
                 dim_out=None,
                 n_layers=2, 
                 n_neurons = 100, 
                 activation = "relu",
                 output_activation = "linear",
                 censored_left = -np.inf,
                 censored_right = np.inf,
                 optimizer = "Adam",
                 optimizer_kwargs = {},
                 input_scaler = None,
                 shared_input_scaler = False,
                 output_scaler = None,
                 shared_output_scaler = False,
                 show_model_summary=False
                 ):

        # forward model hyperparameters
        self.dim_in = dim_in
        self.dim_out = dim_out
        
        self.output_activation = output_activation
        self.n_layers = n_layers
        self.n_neurons = n_neurons
        self.activation = activation
        
        # censored_left and censored_right should be arrays of shape (dim_out, 1)
        if isinstance(censored_left, (list, tuple, np.ndarray)):
            self.censored_left = np.reshape(censored_left, (1,-1)) # if it's an array with length dim out
        else:
            self.censored_left = np.zeros((1, self.dim_out)) + censored_left # if it's a float
        
        if isinstance(censored_right, (list, tuple, np.ndarray)):
            self.censored_right = np.reshape(censored_right, (1,-1))
        else:
            self.censored_right = np.zeros((1, self.dim_out)) + censored_right
        
        # training
        if isinstance(optimizer, str):
            self.optimizer = getattr(tf.keras.optimizers, optimizer)(**optimizer_kwargs)
        else:
            self.optimizer = optimizer

        # input and output scaling
        self.input_scaler = input_scaler
        self.shared_input_scaler = shared_input_scaler
        self.output_scaler = output_scaler
        self.shared_output_scaler = shared_output_scaler

        self.show_model_summary = show_model_summary
        
        self._expand_y_dim = False # internal flag used for preparing y arrays in correct shape
 
        
    def fit(self,
            x,
            y,
            batch_size=32,
            epochs=1,
            verbose=0,
            callbacks=None,
            validation_split=0.0,
            x_val=None,
            y_val=None,
            shuffle=True,
            sample_weight=None,
            initial_epoch=0,
            steps_per_epoch=None,
            validation_steps=None,
            validation_batch_size=None,
            early_stopping=False,
            patience=20,
            restore_best_weights=True,
            plot_learning_curve=True,
            validation_freq=1,
            max_queue_size=10,
            workers=1,
            use_multiprocessing=False):
            
        """Trains the model for a fixed number of epochs (iterations on a dataset).
        
        Args:
        x: Input data. It could be:
          - A Numpy array (or array-like), or a list of arrays
            (in case the model has multiple inputs).
          - A TensorFlow tensor, or a list of tensors
            (in case the model has multiple inputs).
          - A dict mapping input names to the corresponding array/tensors,
            if the model has named inputs.
          - A `tf.data` dataset. Should return a tuple
            of either `(inputs, targets)` or
            `(inputs, targets, sample_weights)`.
          - A generator or `keras.utils.Sequence` returning `(inputs, targets)`
            or `(inputs, targets, sample_weights)`.
          - A `tf.keras.utils.experimental.DatasetCreator`, which wraps a
            callable that takes a single argument of type
            `tf.distribute.InputContext`, and returns a `tf.data.Dataset`.
            `DatasetCreator` should be used when users prefer to specify the
            per-replica batching and sharding logic for the `Dataset`.
            See `tf.keras.utils.experimental.DatasetCreator` doc for more
            information.
          A more detailed description of unpacking behavior for iterator types
          (Dataset, generator, Sequence) is given below. If using
          `tf.distribute.experimental.ParameterServerStrategy`, only
          `DatasetCreator` type is supported for `x`.
        y: Target data. Like the input data `x`,
          it could be either Numpy array(s) or TensorFlow tensor(s).
          It should be consistent with `x` (you cannot have Numpy inputs and
          tensor targets, or inversely). If `x` is a dataset, generator,
          or `keras.utils.Sequence` instance, `y` should
          not be specified (since targets will be obtained from `x`).
        batch_size: Integer or `None`.
            Number of samples per gradient update.
            If unspecified, `batch_size` will default to 32.
            Do not specify the `batch_size` if your data is in the
            form of datasets, generators, or `keras.utils.Sequence` instances
            (since they generate batches).
        epochs: Integer. Number of epochs to train the model.
            An epoch is an iteration over the entire `x` and `y`
            data provided
            (unless the `steps_per_epoch` flag is set to
            something other than None).
            Note that in conjunction with `initial_epoch`,
            `epochs` is to be understood as "final epoch".
            The model is not trained for a number of iterations
            given by `epochs`, but merely until the epoch
            of index `epochs` is reached.
        verbose: 'auto', 0, 1, or 2. Verbosity mode.
            0 = silent, 1 = progress bar, 2 = one line per epoch.
            'auto' defaults to 1 for most cases, but 2 when used with
            `ParameterServerStrategy`. Note that the progress bar is not
            particularly useful when logged to a file, so verbose=2 is
            recommended when not running interactively (eg, in a production
            environment).
        callbacks: List of `keras.callbacks.Callback` instances.
            List of callbacks to apply during training.
            See `tf.keras.callbacks`. Note `tf.keras.callbacks.ProgbarLogger`
            and `tf.keras.callbacks.History` callbacks are created automatically
            and need not be passed into `model.fit`.
            `tf.keras.callbacks.ProgbarLogger` is created or not based on
            `verbose` argument to `model.fit`.
            Callbacks with batch-level calls are currently unsupported with
            `tf.distribute.experimental.ParameterServerStrategy`, and users are
            advised to implement epoch-level calls instead with an appropriate
            `steps_per_epoch` value.
        validation_split: Float between 0 and 1.
            Fraction of the training data to be used as validation data.
            The model will set apart this fraction of the training data,
            will not train on it, and will evaluate
            the loss and any model metrics
            on this data at the end of each epoch.
            The validation data is selected from the last samples
            in the `x` and `y` data provided, before shuffling. This argument is
            not supported when `x` is a dataset, generator or
            `keras.utils.Sequence` instance.
            If both `validation_data` and `validation_split` are provided,
            `validation_data` will override `validation_split`.
            `validation_split` is not yet supported with
            `tf.distribute.experimental.ParameterServerStrategy`.
        validation_data: Data on which to evaluate
            the loss and any model metrics at the end of each epoch.
            The model will not be trained on this data. Thus, note the fact
            that the validation loss of data provided using `validation_split`
            or `validation_data` is not affected by regularization layers like
            noise and dropout.
            `validation_data` will override `validation_split`.
            `validation_data` could be:
              - A tuple `(x_val, y_val)` of Numpy arrays or tensors.
              - A tuple `(x_val, y_val, val_sample_weights)` of NumPy arrays.
              - A `tf.data.Dataset`.
              - A Python generator or `keras.utils.Sequence` returning
              `(inputs, targets)` or `(inputs, targets, sample_weights)`.
            `validation_data` is not yet supported with
            `tf.distribute.experimental.ParameterServerStrategy`.
        shuffle: Boolean (whether to shuffle the training data
            before each epoch) or str (for 'batch'). This argument is ignored
            when `x` is a generator or an object of tf.data.Dataset.
            'batch' is a special option for dealing
            with the limitations of HDF5 data; it shuffles in batch-sized
            chunks. Has no effect when `steps_per_epoch` is not `None`.
        class_weight: Optional dictionary mapping class indices (integers)
            to a weight (float) value, used for weighting the loss function
            (during training only).
            This can be useful to tell the model to
            "pay more attention" to samples from
            an under-represented class.
        sample_weight: Optional Numpy array of weights for
            the training samples, used for weighting the loss function
            (during training only). You can either pass a flat (1D)
            Numpy array with the same length as the input samples
            (1:1 mapping between weights and samples),
            or in the case of temporal data,
            you can pass a 2D array with shape
            `(samples, sequence_length)`,
            to apply a different weight to every timestep of every sample. This
            argument is not supported when `x` is a dataset, generator, or
           `keras.utils.Sequence` instance, instead provide the sample_weights
            as the third element of `x`.
        initial_epoch: Integer.
            Epoch at which to start training
            (useful for resuming a previous training run).
        steps_per_epoch: Integer or `None`.
            Total number of steps (batches of samples)
            before declaring one epoch finished and starting the
            next epoch. When training with input tensors such as
            TensorFlow data tensors, the default `None` is equal to
            the number of samples in your dataset divided by
            the batch size, or 1 if that cannot be determined. If x is a
            `tf.data` dataset, and 'steps_per_epoch'
            is None, the epoch will run until the input dataset is exhausted.
            When passing an infinitely repeating dataset, you must specify the
            `steps_per_epoch` argument. If `steps_per_epoch=-1` the training
            will run indefinitely with an infinitely repeating dataset.
            This argument is not supported with array inputs.
            When using `tf.distribute.experimental.ParameterServerStrategy`:
              * `steps_per_epoch=None` is not supported.
        validation_steps: Only relevant if `validation_data` is provided and
            is a `tf.data` dataset. Total number of steps (batches of
            samples) to draw before stopping when performing validation
            at the end of every epoch. If 'validation_steps' is None, validation
            will run until the `validation_data` dataset is exhausted. In the
            case of an infinitely repeated dataset, it will run into an
            infinite loop. If 'validation_steps' is specified and only part of
            the dataset will be consumed, the evaluation will start from the
            beginning of the dataset at each epoch. This ensures that the same
            validation samples are used every time.
        validation_batch_size: Integer or `None`.
            Number of samples per validation batch.
            If unspecified, will default to `batch_size`.
            Do not specify the `validation_batch_size` if your data is in the
            form of datasets, generators, or `keras.utils.Sequence` instances
            (since they generate batches).
        validation_freq: Only relevant if validation data is provided. Integer
            or `collections.abc.Container` instance (e.g. list, tuple, etc.).
            If an integer, specifies how many training epochs to run before a
            new validation run is performed, e.g. `validation_freq=2` runs
            validation every 2 epochs. If a Container, specifies the epochs on
            which to run validation, e.g. `validation_freq=[1, 2, 10]` runs
            validation at the end of the 1st, 2nd, and 10th epochs.
        max_queue_size: Integer. Used for generator or `keras.utils.Sequence`
            input only. Maximum size for the generator queue.
            If unspecified, `max_queue_size` will default to 10.
        workers: Integer. Used for generator or `keras.utils.Sequence` input
            only. Maximum number of processes to spin up
            when using process-based threading. If unspecified, `workers`
            will default to 1.
        use_multiprocessing: Boolean. Used for generator or
            `keras.utils.Sequence` input only. If `True`, use process-based
            threading. If unspecified, `use_multiprocessing` will default to
            `False`. Note that because this implementation relies on
            multiprocessing, you should not pass non-picklable arguments to
            the generator as they can't be passed easily to children processes.
        plot_learning_curve=False: Boolean. If 'True' will plot learning curves
            after training is over.
        """
            
        self._fit_x_scaler(x)
        self._fit_y_scaler(y)
        x_, y_, x_val_, y_val_ = self._prepare_training_data(x, y, x_val, y_val)
        
        if self.output_scaler is not None:
            self.censored_left = self._scale_y(self.censored_left)
            self.censored_right = self._scale_y(self.censored_right)
            self.model = self._build_model() # necessary as these are hyper parameters of the model
        
        if early_stopping:
            es_clb = tf.keras.callbacks.EarlyStopping(patience=patience, verbose=verbose, mode="min", restore_best_weights=restore_best_weights)
            if callbacks is None:
                callbacks = es_clb
            elif isinstance(callbacks, (list, tuple, np.ndarray)):
                callbacks = [*callbacks, es_clb]
            else:
                callbacks = [callbacks, es_clb]
            
        # run training
        #print("\nTraining model... \n")
        self.loss_dict = self.model.fit(x=x_, 
                                        y=y_,
                                        batch_size=batch_size, 
                                        epochs=epochs,
                                        verbose=verbose,
                                        callbacks=callbacks,
                                        validation_split=validation_split,
                                        validation_data=(x_val_, y_val_),
                                        shuffle=shuffle,
                                        sample_weight=sample_weight,
                                        initial_epoch=initial_epoch,
                                        steps_per_epoch=steps_per_epoch,
                                        validation_steps=validation_steps,
                                        validation_batch_size=validation_batch_size,
                                        validation_freq=validation_freq,
                                        max_queue_size=max_queue_size,
                                        workers=workers,
                                        use_multiprocessing=use_multiprocessing                                           
                                        )
        #print("\nModel training done. \n")

        
        if plot_learning_curve:
            pd.DataFrame(self.loss_dict.history, dtype=float).plot()
                
        return self
    
    
    def _set_model(self, model):
        # a function to supply custom keras model (needs to be compiled and trainable with model.fit())
        self.model = model
    
    
    def _prepare_training_data(self, x, y, x_val=None, y_val=None):
        # prepare x
        
        x = self._scale_x(x)
        y = self._scale_y(y)
        if self._expand_y_dim:
            y = np.expand_dims(y, axis=-1)        

        if x_val is not None:
            x_val = self._scale_x(x_val)
            y_val = self._scale_y(y_val)
            if self._expand_y_dim:
                y_val = np.expand_dims(y_val, axis=-1)

        return x, y, x_val, y_val
        

    def _fit_x_scaler(self,x):
        # assumes x has shape (N, n_features)
        if self.input_scaler is not None:
            # if a string use one of these
            if isinstance(self.input_scaler,str):
                if self.input_scaler == "MinMax":
                    self.scaler_x = MinMaxScaler()
                elif self.input_scaler == "Standard":
                    self.scaler_x = StandardScaler()
                elif self.input_scaler == "QuantileUniform":
                    self.scaler_x = QuantileTransformer(output_distribution="uniform")
                elif self.input_scaler == "QuantileNormal":
                    self.scaler_x = QuantileTransformer(output_distribution="normal")
                else:
                    raise ValueError(f"Unknown input_scaler keyword: {self.input_scaler}")
            # if not a string we assume that the passed object is a transormation 
            # with a fit(), transform(), and inverse_transform() method (e.g. a log or logit transformer)
            else:
                self.scaler_x = self.input_scaler
            
            self.scaler_x.fit(x)
        
        else:
            pass # if no scaler this function does nothing


    def _scale_x(self, x):
        if self.input_scaler is not None:
            return self.scaler_x.transform(x)
        else:
            return x


    def _fit_y_scaler(self,y):
        
        if self.output_scaler is not None:
            if isinstance(self.output_scaler, str):
                if self.output_scaler == "MinMax":
                    self.scaler_y = MinMaxScaler()
                elif self.output_scaler == "Standard":
                    self.scaler_y = StandardScaler()
                elif self.output_scaler == "QuantileUniform":
                    self.scaler_y = QuantileTransformer(output_distribution="uniform")
                elif self.output_scaler == "QuantileNormal":
                    self.scaler_y = QuantileTransformer(output_distribution="normal")
                else:
                    raise ValueError(f"Unknown output_scaler keyword: {self.output_scaler}")            
            else:
                # if not a string we assume that the passed instance is a transormation 
                # with a fit(), transform(), and inverse_transform() method (e.g. a log or logit transformer)
                self.scaler_y = self.output_scaler
            
            if self.shared_output_scaler is False:
                self.scaler_y.fit(y)
            elif self.shared_output_scaler is True:
                self.scaler_y.fit(np.reshape(y, (-1, 1)))
        
        else:
            pass


    def _scale_y(self, y):
        if self.output_scaler is not None:
            if self.shared_output_scaler is False:
                return self.scaler_y.transform(y)
            elif self.shared_output_scaler is True:
                return np.reshape(self.scaler_y.transform(np.reshape(y, (-1, 1))), (y.shape))
        else:
            return y
        
    
    def _rescale_y(self, y):
        # assumes an array of (n_samples, dim_out)
        if self.output_scaler is not None:
            if self.shared_output_scaler is False:
                return self.scaler_y.inverse_transform(y)
            elif self.shared_output_scaler is True:
                return np.reshape(self.scaler_y.inverse_transform(np.reshape(y, (-1, 1))), (y.shape))
        else:
            return y

    def _rescale_y_samples(self, y):
        # assumes an array of (n_samples, dim_out, n_samples)
        if self.output_scaler is not None:
            D = y.shape[1]
            S = y.shape[2]
            y_ = np.reshape(np.transpose(y, (0,2,1)), (-1, D)) # (N,D,S) --> (N*S,D) 
            y_ = self._rescale_y(y_) # rescale 2-dim (N*S,D) array
            y_ = np.transpose(np.reshape(y_, (-1, S, D)), (0,2,1)) #  (N*S,D) --> (N,D,S)
            return y_
        else:
            return y


################ Class for all models that use univariate margins + copula style ################
class MarginsAndCopulaModel(MVPRegModel):
    def __init__(self,
                 copula_type = "independence",
                 vine_structure = None,
                 pair_copula_families = "nonparametric",
                 **kwargs):
        
        super().__init__(**kwargs)

        if copula_type in ("d-vine", "c-vine") and vine_structure is None:
            raise ValueError(f"'vine_structure' is {vine_structure}. If copula type is 'd-vine' or 'c-vine', 'vine_structure' must be specified by an iterable of length {self.dim_out} (e.g. [1,2,3]).")
        
        self.copula_type = copula_type
        self.vine_structure = vine_structure
        self.pair_copula_families = pair_copula_families

    def fit(self, x, y, fit_copula_model=True, **kwargs):
        super().fit(x, y, **kwargs)
        
        if fit_copula_model:
            #print("\nFitting copula model...")
            self.fit_copula(x, y)
            #print("Done.\n")
        return self
    
    def simulate_copula(self, n_samples=1):
        return np.clip(self.copula.simulate(n_samples), 0.0001, 0.9999)


    def fit_copula(self, x, y):
        pseudo_obs = self.cdf(x, y) # obtain pseudo observations
        
        # ensure that marginals of pseudo_obs are uniform
        self.pseudo_obs_uniform = np.zeros_like(pseudo_obs)
        for d in range(pseudo_obs.shape[1]):
            self.pseudo_obs_uniform[:,d] = (2*rank_data_random_tiebreaker(pseudo_obs[:,d])-1)/(2*pseudo_obs.shape[0]) # assign ordinal ranks and break ties at random
        
        self.copula = self._get_copula(self.pseudo_obs_uniform) # fit copula model on uniform pobs

        return self


    def _get_copula(self, u):
        
        u = np.clip(u, 1e-8, 1.0-1e-8)
        
        if self.copula_type == "gaussian":
            copula = GaussianCopula().fit(u)
        
        elif self.copula_type == "independence":
            copula = IndependenceCopula(dim=u.shape[1])
            
        elif self.copula_type == "schaake":
            copula = SchaakeShuffle().fit(u)
        
        elif self.copula_type in ("r-vine", "d-vine", "c-vine"):
            copula = VineCopula(pair_copula_families=self.pair_copula_families, vine_structure=self.vine_structure, vine_type=self.copula_type).fit(u)
        
        else:
            raise ValueError(f"Copula type {self.copula_type} is unknown. Must be one of: independence, schaake, gaussian, r-vine, c-vine, d-vine.")
        
        return copula
        