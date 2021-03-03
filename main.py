import numpy as np
import pandas as pd
import tensorflow as tf
import IPython
import kerastuner as kt
from utils import get_clean_data
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, normalize
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score,
    accuracy_score,
)
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from keras import optimizers, utils


def create_prelim_dataset(X, Y, look_back):
    """
    Reshape feature vector and output vector.

    Inputs:
        X: The feature vector.
        Y: The output vector.
        look_back: The number of previous minutes' data to include in each vector of the output matrices.

    Returns:
        np.array(dataX): A numpy array of vectors each with 2 * look_back + 2 preprocessed values;
            The current hour
            The current minute
            look_back standard deviations for the past look_back minutes
            look_back average prices for the past look_back minutes
            look_back trend indicators for the past look_back minutes
        np.array(dataY):
    """
    dataX, dataY = [], []
    # Start at the look_back index, because every minute prior to this index doesn't have enough data
    # for a complete feature vector.
    for i in range(look_back, len(X)):
        feature_vector = []
        # Add the relevant data to the feature vector starting from look_back minutes
        # prior to the current minute.
        for j in range(look_back + 1):
            for element in X[(i - look_back) + j]:
                feature_vector.append(element)
        dataX.append([feature_vector])
        dataY.append(Y[i])
    return np.array(dataX), np.array(dataY)


def create_features(X, look_back):
    """
    Reshape feature vector.

    Inputs:
        X: The feature vector.
        look_back: The number of previous minutes' data to include in each vector of the output matrices.

    Returns:
        np.array(dataX): A numpy array of vectors each with 2 * look_back + 2 preprocessed values;
            The current hour
            The current minute
            look_back standard deviations for the past look_back minutes
            look_back average prices for the past look_back minutes
            look_back trend indicators for the past look_back minutes
    """
    dataX = []
    # Start at the look_back index, because every minute prior to this index doesn't have enough data
    # for a complete feature vector.
    for i in range(look_back, len(X)):
        feature_vector = []
        # Add the relevant data to the feature vector starting from look_back minutes
        # prior to the current minute.
        for j in range(look_back + 1):
            # We only want the minute and hour of the current minute, not the look_back minutes.
            for element in X[(i - look_back) + j]:
                feature_vector.append(element)
        dataX.append([feature_vector])
    return np.array(dataX)


def get_model():
    """
    Use Keras Tuner to tune the hyperparameters.
    """
    # Use the mirrored strategy to take advantage of system GPUs for NN training.
    strategy = tf.distribute.MirroredStrategy()
    # Andrew Ng recommends 512 batch size for mini batching.
    BATCH_SIZE_PER_REPLICA = 512
    BATCH_SIZE = 512 * strategy.num_replicas_in_sync

    # Use the hyperband algorithm to test a large number of models using different hyperparameters and
    # pick the model with the best validation set accuracy. Use 500 epochs, 3 iterations, and the mirrored GPU strategy.
    tuner = kt.Hyperband(
        model_builder,
        executions_per_trial=1,
        objective="val_accuracy",
        max_epochs=500,
        factor=2,
        hyperband_iterations=3,
        distribution_strategy=strategy,
        tune_new_entries=True,
        allow_new_entries=True,
        directory="DNN_Hypersearch",
        project_name="multi-layer",
    )
    tuner.search_space_summary()
    # Use the early stopping callback after 20 iterations.
    tuner.search(
        trainX,
        trainY,
        epochs=300,
        batch_size=BATCH_SIZE,
        validation_data=(validX, validY),
        callbacks=[ClearTrainingOutput(), keras.callbacks.EarlyStopping(patience=20)],
    )
    # Output the best model and save it.
    model = tuner.get_best_models(1)[0]
    model.save("./saved_model/my_model")


def model_builder(hp):
    # TODO: Iterate on amount of training data, type of training data look_back, buffer.
    # Consider other metrics (F1) https://datascience.stackexchange.com/questions/45165/how-to-get-accuracy-f1-precision-and-recall-for-a-keras-model

    # Current params being tuned:
    #   1. Learning_rate
    #   2. Number of layers (maybe try type of layers)
    #   3. Number of hidden units per layer
    #   4. Regularization parameters
    #       Dropout: https://keras.io/api/layers/regularization_layers/dropout/
    #       L2: https://keras.io/api/layers/regularizers/
    #   5. Exponential/InverseTime Decay

    # Only change if literature indicates it will help:
    #   1. Optimizer (current: Adam) https://keras.io/api/optimizers/ https://keras.io/api/optimizers/adam/
    #   2. Activation functions (current: relu for hidden, sigmoid for output) https://keras.io/api/layers/activations/
    #   3. Loss function (current: binary crossentropy) https://keras.io/api/losses/probabilistic_losses/#sparsecategoricalcrossentropy-class
    #   4. Initializer (current: He) https://www.tensorflow.org/api_docs/python/tf/keras/initializers https://keras.io/api/layers/initializers/
    #   5. Mini-batch size (current: 512)

    # Try tuning after everything else:
    #   1. Batch normalization hyperparameters https://keras.io/api/layers/normalization_layers/batch_normalization/
    #   2. Adam hyperparameters
    #   2. Early Stopping
    #   3. Switching from adam to SGD as training progresses
    #   4. Consider unrolling

    # Each vector is 1 minute x 2 * look_back + 2
    inputs = keras.Input(shape=(1, 6))
    initializer = keras.initializers.HeNormal()
    # Keras tuner adjusts the nodes per layer in each model, where the minimum possible nodes is 32, and the max is 512 with a 32 step.
    # Each layer is adjusted indepently, so one layer could have 32 nodes and another 64.
    hp_units = hp.Int("units", min_value=32, max_value=512, step=32)
    # x is the input layer, which is linear and uses the relu activation function.
    x = keras.layers.Dense(
        units=hp_units, activation="relu", kernel_initializer=initializer
    )(inputs)
    # Keras Tuner creates models with between 2 and 10 layers.
    for i in range(hp.Int("num_layers", 2, 10)):
        # In each layer, the same initializaer is used, but the l2 min/max values for the kernel, bias, and activity regularizers are
        # tuned by Keras Tuner.
        x = layers.Dense(
            units=hp.Int("units_" + str(i), min_value=32, max_value=512, step=32),
            kernel_initializer=initializer,
            kernel_regularizer=regularizers.l2(
                hp.Float("kernel_l2", max_value=1e-2, min_value=1e-5, sampling="log")
            ),
            bias_regularizer=regularizers.l2(
                hp.Float("bias_l2", max_value=1e-2, min_value=1e-5, sampling="log")
            ),
            activity_regularizer=regularizers.l2(
                hp.Float("activity_l2", max_value=1e-2, min_value=1e-5, sampling="log")
            ),
        )(x)
        # BatchNormalization is used on every layer.
        x = keras.layers.BatchNormalization()(x)
        # Every layer uses the relu activation function.
        x = layers.Activation("relu")(x)
        # The dropout value is tuned for each layer.
        x = layers.Dropout(
            hp.Float(
                "dropout_" + str(i), max_value=0.7, min_value=0.1, sampling="linear"
            )
        )(x)

    # The sigmoid output function is used.
    outputs = keras.layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    # The initial learning rate and decay rate are tuned by Keras Tuner.
    lr_schedule = keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=hp.Float(
            "initial_learning_rate", max_value=1, min_value=1e-4, sampling="log"
        ),
        decay_steps=23000 / 512 * 1000,
        decay_rate=hp.Float(
            "decay_rate", max_value=1, min_value=0.1, sampling="linear"
        ),
    )
    # The Adam optimizer is used.
    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
    # The model is trained using accuracy, and the loss function is binary crossentropy.
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
    return model


if __name__ == "__main__":
    features = ["Pseudo-Log-Return", "Trend Indicator"]
    label = ["Direction"]

    # Get each set of data.
    df_train, df_valid, df_test = get_clean_data()
    Xtrain = df_train[features].values
    Ytrain = df_train[label].values.ravel()
    Xvalid = df_valid[features].values
    Yvalid = df_valid[label].values.ravel()
    Xtest = df_test[features].values
    Ytest = df_test[label].values.ravel()

    # Store the data for each set in the corresponding feature and output vectors.
    # Preprocess the data. https://scikit-learn.org/stable/modules/preprocessing.html
    scale = MinMaxScaler(feature_range=(0, 1))
    Xtrain = scale.fit_transform(Xtrain)
    Ytrain = scale.fit_transform(Ytrain.reshape(-1, 1))
    Xvalid = scale.fit_transform(Xvalid)
    Yvalid = scale.fit_transform(Yvalid.reshape(-1, 1))
    Xtest = scale.fit_transform(Xtest)
    Ytest = scale.fit_transform(Ytest.reshape(-1, 1))

    # Reshape the data.
    look_back = 2
    trainX, trainY = create_prelim_dataset(Xtrain, Ytrain, look_back)
    validX, validY = create_prelim_dataset(Xvalid, Yvalid, look_back)
    testX, testY = create_prelim_dataset(Xtest, Ytest, look_back)

    # Retrieve the model.
    # get_model()
    model = tf.keras.models.load_model("./saved_model/my_model")

    # Use the model to make predictions on the training data.
    lstm_train_pred = model.predict(trainX)
    lstm_train_pred_df = pd.DataFrame(
        lstm_train_pred.reshape(trainY.shape[0], trainY.shape[1])
    )
    # Output all predictions to a CSV file.
    lstm_train_pred_df.to_csv("./CleanData/dnn_train_pred.csv", index=False)
    lstm_train_pred = pd.read_csv(
        "./CleanData/dnn_train_pred.csv", error_bad_lines=False
    )

    # Use the model to make predictions on the validation set.
    lstm_valid_pred = model.predict(validX)
    valid_accuracy = model.evaluate(validX, validY)
    # Report the accuracy and the AUC score.
    print("Accuracy DNN on validation " + str(valid_accuracy[1]))
    y_true = Yvalid[look_back:].ravel()
    print(
        "AUC Score of DNN on valid: ",
        roc_auc_score(
            np.array(y_true),
            lstm_valid_pred.reshape(lstm_valid_pred.shape[0], lstm_valid_pred.shape[1]),
        ),
    )
    lstm_valid_pred_df = pd.DataFrame(
        lstm_valid_pred.reshape(validY.shape[0], validY.shape[1])
    )
    # Output all predictions to a CSV file.
    lstm_valid_pred_df.to_csv("./CleanData/dnn_valid_pred.csv", index=False)
    lstm_valid_pred = pd.read_csv(
        "./CleanData/dnn_valid_pred.csv", error_bad_lines=False
    )

    # Use the model to makepredictions on the test set.
    lstm_test_pred = model.predict(testX)
    test_accuracy = model.evaluate(testX, testY)
    # Report the accuracy.
    print("Accuracy DNN on test: " + str(test_accuracy[1]))
    # Calculate and output AUC score.
    y_true = Ytest[look_back:].ravel()
    print(
        "AUC Score of DNN on test: ",
        roc_auc_score(
            np.array(y_true),
            lstm_test_pred.reshape(lstm_test_pred.shape[0], lstm_test_pred.shape[1]),
        ),
    )
    lstm_test_pred_df = pd.DataFrame(
        lstm_test_pred.reshape(testY.shape[0], testY.shape[1])
    )
    # Output all predictions to a CSV file.
    lstm_test_pred_df.to_csv("./CleanData/dnn_test_pred.csv", index=False)
    lstm_test_pred = pd.read_csv("./CleanData/dnn_test_pred.csv", error_bad_lines=False)
