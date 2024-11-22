import re
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import chardet
import pandas as pd

class HuggingFaceModel:
    def __init__(self, model_name="jcortizba/modelo1"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        # Mapeo de etiquetas
        self.label_mapping = {0: "ejecutar", 1: "cancelar"}

    def predict(self, text):
        # Validar entrada antes de predecir
        if not self.validate_input(text):
            return "Entrada inválida. Por favor, ingrese un texto coherente."
        
        inputs = self.tokenizer(text, return_tensors="pt")
        outputs = self.model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_index = torch.argmax(probabilities, dim=1).item()
        predicted_label = self.label_mapping[predicted_index]
        return predicted_label

    def validate_input(self, text):
        """Valida que el texto sea coherente y legible."""
        if len(text.strip()) < 3:  # Rechazar entradas cortas
            return False
        if not re.match(r"^[a-zA-Z0-9ñÑ\s.,!?]+$", text):  # Validar caracteres
            return False
        if text.strip().isdigit():  # Rechazar entradas puramente numéricas
            return False
        return True

# Función para procesar un archivo .csv
def process_file(file):
    """
    Procesa un archivo CSV, predice sobre una columna específica y retorna el resultado.
    """
    try:
        # Detectar la codificación del archivo
        with open(file.name, 'rb') as f:
            raw_data = f.read()
            result = chardet.detect(raw_data)
            encoding = result['encoding']  # Obtén la codificación detectada
        
        # Leer el archivo con la codificación detectada
        df = pd.read_csv(file.name, encoding=encoding)
        
        # Verificar si la columna para predecir existe
        if 'Descripción de la orden' not in df.columns:
            return "Error: El archivo no tiene una columna llamada 'Descripción de la orden'.", None
        
        # Instanciar el modelo
        model = HuggingFaceModel()
        
        # Aplicar predicciones en la columna 'Descripción de la orden'
        df['out'] = df['Descripción de la orden'].apply(model.predict)
        
        # Guardar el archivo con resultados
        output_file = "resultado_predicciones.csv"
        df.to_csv(output_file, index=False, encoding='utf-8')
        
        return "Archivo procesado exitosamente.", output_file
    except Exception as e:
        return f"Error al procesar el archivo: {e}", None

# Función para pruebas individuales
def predict_individual_text(text):
    """
    Predice la etiqueta para un texto individual.
    """
    model = HuggingFaceModel()
    return model.predict(text)

# Interfaz Gradio
def main():
    with gr.Blocks() as demo:
        gr.Markdown("# Modelo Hugging Face - Predicción Individual y desde CSV")
        
        # Entrada para pruebas individuales
        gr.Markdown("### Predicción Individual")
        input_text = gr.Textbox(label="Ingrese texto para predecir")
        output_label = gr.Textbox(label="Etiqueta de predicción o mensaje de error", interactive=False)
        predict_button = gr.Button("Predecir")
        
        # Entrada para procesamiento de archivos
        gr.Markdown("### Predicción desde Archivo CSV")
        file_input = gr.File(label="Sube un archivo .csv", file_types=[".csv"])
        output_message = gr.Textbox(label="Estado del proceso", interactive=False)
        result_file = gr.File(label="Archivo con predicciones")
        submit_button = gr.Button("Procesar archivo")
        
        # Vinculación de funciones
        predict_button.click(
            fn=predict_individual_text,
            inputs=[input_text],
            outputs=[output_label],
        )
        
        def handle_process(file):
            message, result_path = process_file(file)
            return message, result_path

        submit_button.click(
            fn=handle_process,
            inputs=[file_input],
            outputs=[output_message, result_file],
        )
    
    demo.launch()

if __name__ == "__main__":
    main()
