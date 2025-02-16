import random
import tensorflow as tf
import numpy as np


class ModelManager:
    def __init__(self):
        self.loss_histories = []
        self.learning_rates = []
        self.W_initials = []
        self.W_finals = []
        self.y_preds = []
        self.kfold_results = []

    def normalizar_datos(self, X_values, y_values):
        X_min, X_max = X_values.min(), X_values.max()
        y_min, y_max = y_values.min(), y_values.max()
        X_norm = (X_values - X_min) / (X_max - X_min)
        y_norm = (y_values - y_min) / (y_max - y_min)
        return X_norm, y_norm, X_min, X_max, y_min, y_max

    def crear_modelo(self, input_dim, lr):
        W = tf.Variable(tf.random.normal(shape=(input_dim, 1), dtype=tf.float32))
        b = tf.Variable(tf.zeros(shape=(1,)), dtype=tf.float32)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        return W, b, optimizer

    def entrenar_fold(self, X_train, y_train, W, b, optimizer, epochs=100):
        loss_history = []
        for epoch in range(epochs):
            with tf.GradientTape() as tape:
                y_pred_norm = tf.matmul(X_train, W) + b
                loss = tf.reduce_mean(tf.square(y_train - y_pred_norm))
            gradients = tape.gradient(loss, [W, b])
            optimizer.apply_gradients(zip(gradients, [W, b]))
            loss_history.append(loss.numpy())
        return loss_history, W, b

    def predecir(self, X, W, b, y_min, y_max):
        y_pred_norm = tf.matmul(X, W) + b
        y_pred_desnorm = y_pred_norm.numpy() * (y_max - y_min) + y_min
        return y_pred_desnorm

    def procesar_datos(self, data, label_status):
        if data is None:
            label_status.config(text="Cargue un archivo CSV primero", foreground="red")
            return

        try:
            X_values = data.iloc[:, :-1].values
            y_values = data.iloc[:, -1].values.reshape(-1, 1)
        except IndexError:
            label_status.config(text="Formato de CSV inválido", foreground="red")
            return

        X_norm, y_norm, X_min, X_max, y_min, y_max = self.normalizar_datos(X_values, y_values)
        X = tf.convert_to_tensor(X_norm, dtype=tf.float32)
        y = tf.convert_to_tensor(y_norm, dtype=tf.float32)

        input_dim = X.shape[1]
        epochs = 100
        self.learning_rates = [random.uniform(0, 2) for _ in range(10)]
        self.learning_rates.append(1.0)

        self.loss_histories = []
        self.W_initials = []
        self.W_finals = []
        self.y_preds = []
        self.kfold_results = []

        k = 3
        fold_size = len(X) // k

        for lr in self.learning_rates:
            W_initials_lr = []
            W_finals_lr = []
            loss_histories_lr = []
            y_preds_lr = []
            fold_losses = []

            for i in range(k):
                val_indices = tf.convert_to_tensor(range(i * fold_size, (i + 1) * fold_size), dtype=tf.int32)
                X_val = tf.gather(X, val_indices)
                y_val = tf.gather(y, val_indices)

                train_indices = tf.concat([tf.range(0, i * fold_size), tf.range((i + 1) * fold_size, len(X))], axis=0)
                X_train = tf.gather(X, train_indices)
                y_train = tf.gather(y, train_indices)

                W, b, optimizer = self.crear_modelo(input_dim, lr)
                W_initials_lr.append(W.numpy().flatten().tolist())

                loss_history, W, b = self.entrenar_fold(X_train, y_train, W, b, optimizer, epochs)
                loss_histories_lr.append(loss_history)
                W_finals_lr.append(W.numpy().flatten().tolist())

                y_pred_desnorm = self.predecir(X_val, W, b, y_min, y_max)
                y_preds_lr.append(y_pred_desnorm)

                fold_losses.append(loss_history[-1])

            avg_loss_history = np.mean(np.array(loss_histories_lr), axis=0)
            self.loss_histories.append(avg_loss_history.tolist())

            self.W_initials.append(W_initials_lr[0])
            self.W_finals.append(W_finals_lr[0])

            W, b, optimizer = self.crear_modelo(input_dim, lr)
            loss_history, W, b = self.entrenar_fold(X, y, W, b, optimizer, epochs)
            y_pred_desnorm = self.predecir(X, W, b, y_min, y_max)
            self.y_preds.append(y_pred_desnorm)

            n_train = len(X_train)
            n_val = len(X_val)
            error_entrenamiento = loss_history[-1]
            error_validacion = np.mean(fold_losses)
            error_combinado = (n_train * error_entrenamiento + n_val * error_validacion) / (n_train + n_val)

            self.kfold_results.append((lr, error_entrenamiento, error_validacion, error_combinado, epochs))

        self.y_real_desnorm = y.numpy() * (y_max - y_min) + y_min
