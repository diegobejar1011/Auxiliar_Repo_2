import tkinter as tk
from tkinter import ttk
from DataManager import DataManager
from ModelManager import ModelManager

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

        # Agregar los resultados a la tabla
        for i, (lr, error_entrenamiento, error_validacion, error_combinado, epochs) in enumerate(self.model_manager.kfold_results):
            W0 = self.model_manager.W_initials[i]
            Wf = self.model_manager.W_finals[i]

            w0_str = "[" + ", ".join([f"{val:.4f}" for val in W0]) + "]"
            wf_str = "[" + ", ".join([f"{val:.4f}" for val in Wf]) + "]"
            self.tree.insert("", "end", values=(f"{lr:.4f}", w0_str, wf_str, f"{error_entrenamiento:.6f}", f"{error_validacion:.6f}", f"{error_combinado:.6f}", epochs))
