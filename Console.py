import random
import tensorflow as tf
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog

class DataManager:
    def __init__(self):
        self.data = None

    def cargar_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("Archivos CSV", "*.csv")])
        return pd.read_csv(file_path, header=None, delimiter=";") if file_path else None

    def iniciar_proceso(self, label_status):
        self.data = self.cargar_csv()
        if self.data is not None:
            label_status.config(text="Datos cargados correctamente", foreground="green")
        else:
            label_status.config(text="No se seleccionó ningún archivo", foreground="red")

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
        self.learning_rates.append(1.0)  # Tasa de aprendizaje adicional predefinida

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

            # Calcular el error combinado
            n_train = len(X_train)
            n_val = len(X_val)
            error_entrenamiento = loss_history[-1]
            error_validacion = np.mean(fold_losses)
            error_combinado = (n_train * error_entrenamiento + n_val * error_validacion) / (n_train + n_val)

            self.kfold_results.append((lr, error_entrenamiento, error_validacion, error_combinado, epochs))

        self.y_real_desnorm = y.numpy() * (y_max - y_min) + y_min

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Panel de Entrenamiento de Datos")
        self.root.geometry("1000x800")
        self.root.minsize(800, 600)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Configurar estilos directamente
        self.configurar_estilos()

        # Inicializar el frame principal
        self.frame_principal = ttk.Frame(self.root, padding=20)
        self.frame_principal.grid(row=0, column=0, sticky="nsew")
        self.frame_principal.columnconfigure(0, weight=1)
        self.frame_principal.rowconfigure(0, weight=0)

        # Inicializar el DataManager
        self.data_manager = DataManager()

        # Inicializar el ModelManager
        self.model_manager = ModelManager()

        # Configurar la interfaz de usuario
        self.setup_ui()

    def configurar_estilos(self):
        """Configura los estilos visuales de la interfaz."""
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('.', background='#f0f0f0', foreground='#333', font=('Segoe UI', 9))
        style.configure('TLabel', padding=6)
        style.configure('TButton', padding=6, relief='raised', borderwidth=2, font=('Segoe UI', 9, 'bold'))
        style.map('TButton', foreground=[('active', '#fff')], background=[('active', '#4CAF50')])
        style.configure('TEntry', padding=5, borderwidth=2)
        style.configure("Treeview", background="#FFFFFF", foreground="#000000", rowheight=25, fieldbackground="#D3D3D3", font=('Segoe UI', 9))
        style.map("Treeview", background=[("selected", "#4CAF50")], foreground=[("selected", "#FFFFFF")])
        style.configure('TLabelframe', padding=10, borderwidth=2)
        style.configure('TLabelframe.Label', font=('Segoe UI', 10, 'bold'))

    def setup_ui(self):
        self.setup_carga_datos()
        self.setup_entrenamiento()
        self.setup_tabla_resultados()

    def setup_carga_datos(self):
        frame_carga = ttk.LabelFrame(self.frame_principal, text="Carga de Datos", padding=10)
        frame_carga.grid(row=1, column=0, sticky="ew", pady=10)
        frame_carga.columnconfigure(0, weight=1)
        frame_carga.columnconfigure(1, weight=2)

        btn_cargar = ttk.Button(frame_carga, text="Cargar CSV", command=self.iniciar_proceso)
        btn_cargar.grid(row=0, column=0, sticky="w", padx=5)

        self.label_status = ttk.Label(frame_carga, text="Seleccione un archivo CSV", foreground="grey")
        self.label_status.grid(row=0, column=1, sticky="w", padx=5)

    def setup_entrenamiento(self):
        frame_entrenamiento = ttk.LabelFrame(self.frame_principal, text="Entrenamiento", padding=10)
        frame_entrenamiento.grid(row=2, column=0, sticky="ew", pady=10)

        btn_entrenar = ttk.Button(frame_entrenamiento, text="Entrenar Modelo", command=self.procesar_datos)
        btn_entrenar.pack(pady=10)

    def setup_tabla_resultados(self):
        frame_tabla = ttk.LabelFrame(self.frame_principal, text="Resultados del Entrenamiento", padding=10)
        frame_tabla.grid(row=3, column=0, sticky="nsew", pady=10)

        self.tree = ttk.Treeview(frame_tabla, columns=("LR", "W0", "Wf", "Error Entrenamiento", "Error Validación", "Error Combinado", "Épocas"), show="headings")
        self.tree.heading("LR", text="Tasa de Aprendizaje")
        self.tree.heading("W0", text="Pesos Iniciales")
        self.tree.heading("Wf", text="Pesos Finales")
        self.tree.heading("Error Entrenamiento", text="Error Entrenamiento")
        self.tree.heading("Error Validación", text="Error Validación")
        self.tree.heading("Error Combinado", text="Error Combinado")
        self.tree.heading("Épocas", text="Épocas")
        self.tree.pack(fill=tk.BOTH, expand=True)

    def iniciar_proceso(self):
        self.data_manager.iniciar_proceso(self.label_status)

    def procesar_datos(self):
        self.model_manager.procesar_datos(self.data_manager.data, self.label_status)
        self.actualizar_tabla()

    def actualizar_tabla(self):
        # Limpiar la tabla antes de agregar nuevos datos
        for row in self.tree.get_children():
            self.tree.delete(row)

        # Imprimir los resultados en la consola
        print("\nResultados del Entrenamiento:")
        print("{:<15} {:<20} {:<20} {:<20} {:<20} {:<20} {:<10}".format(
            "Tasa de Aprendizaje", "Pesos Iniciales", "Pesos Finales", "Error Entrenamiento", "Error Validación", "Error Combinado", "Épocas"
        ))
        print("-" * 130)

        # Agregar los resultados a la tabla y a la consola
        for i, (lr, error_entrenamiento, error_validacion, error_combinado, epochs) in enumerate(self.model_manager.kfold_results):
            W0 = self.model_manager.W_initials[i]
            Wf = self.model_manager.W_finals[i]

            w0_str = "[" + ", ".join([f"{val:.4f}" for val in W0]) + "]"
            wf_str = "[" + ", ".join([f"{val:.4f}" for val in Wf]) + "]"

            # Insertar en la tabla
            self.tree.insert("", "end", values=(f"{lr:.4f}", w0_str, wf_str, f"{error_entrenamiento:.6f}", f"{error_validacion:.6f}", f"{error_combinado:.6f}", epochs))

            # Imprimir en la consola
            print("{:<15.4f} {:<20} {:<20} {:<20.6f} {:<20.6f} {:<20.6f} {:<10}".format(
                lr, w0_str, wf_str, error_entrenamiento, error_validacion, error_combinado, epochs
            ))

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()