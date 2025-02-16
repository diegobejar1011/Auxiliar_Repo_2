import pandas as pd
from tkinter import filedialog

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
