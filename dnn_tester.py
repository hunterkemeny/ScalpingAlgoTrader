    def train_dnn(self, X_train, Y_train, X_val, Y_val):
        """
        
        Inputs:
            X_train: The num_training_minutes x 3 * look_back + 2 input training matrix.
            Y_train: The num_training minutes x 1 output training matrix.
            X_val: The num_validation_minutes x 3 * look_back + 2 input training matrix.
            Y_val: The num_validation_minutes x 1 output validation matrix.
    
        """
        strategy = tf.distribute.MirroredStrategy()
        assert len(X_train) == len(Y_train)
        assert len(X_val) == len(Y_val)
        print(X_train.shape)
        print(Y_train.shape)
        with strategy.scope():
            inputs = keras.Input(shape=(X_train.shape[1], X_train.shape[2]))
            initializer = keras.initializers.HeNormal()
            x = layers.Dense(100, kernel_initializer=initializer, 
                            kernel_regularizer=regularizers.l2(l2=0.1))(inputs)
            x = keras.layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            x = layers.Dropout(0.5)(x)
            x = layers.Dense(100, kernel_initializer=initializer, 
                            kernel_regularizer=regularizers.l2(l2=0.01))(x)
            x = keras.layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            x = layers.Dropout(0.5)(x)
            x = layers.Dense(100, kernel_initializer=initializer, 
                            kernel_regularizer=regularizers.l2(l2=0.01))(x)
            x = keras.layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            x = layers.Dropout(0.5)(x)
            x = layers.Dense(100, kernel_initializer=initializer, 
                            kernel_regularizer=regularizers.l2(l2=0.01))(x)
            x = keras.layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            x = layers.Dropout(0.5)(x)

            
            outputs = layers.Dense(1, activation='sigmoid')(x)
            self.model = keras.Model(inputs=inputs, outputs=outputs)
            self.model.summary()
            # early_stopping = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=20)
            lr_schedule = keras.optimizers.schedules.InverseTimeDecay(initial_learning_rate=0.001,
                                                                        decay_steps=50000,
                                                                        decay_rate=1)
            optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
            BATCH_SIZE_PER_REPLICA = 512
            BATCH_SIZE = 512 * strategy.num_replicas_in_sync
            self.model.compile(loss = 'binary_crossentropy', optimizer = optimizer, metrics = ['accuracy'])
            self.model.fit(X_train, Y_train, epochs = 5000, batch_size = BATCH_SIZE, callbacks = [], validation_data = (X_val, Y_val))