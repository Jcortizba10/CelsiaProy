import re
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import chardet
import pandas as pd

class HuggingFaceModel:
    def __init__(self, model_name="jcortizba/modelo18"):
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
        probabilities_list = probabilities.tolist()

        predicted_index = torch.argmax(probabilities, dim=1).item()
        predicted_label = self.label_mapping[predicted_index]

        return predicted_label
    
    

    def validate_input(self, text):
        """Valida que el texto sea coherente y legible."""
        # Rechazar entradas vacías o con menos de 3 caracteres
        if len(text.strip()) < 3:
            return False
        # Permitir solo letras, números, espacios y signos de puntuación básicos
        if not re.match(r"^[a-zA-Z0-9ñÑ\s.,!?]+$", text):
            return False
        # Verificar que no sea puramente numérico
        if text.strip().isdigit():
            return False
        return True

# Función para procesar un archivo .csv
def process_file(file):
    """
    Carga un archivo CSV, detecta su codificación, realiza predicciones en una columna específica,
    y exporta el archivo con los resultados.
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
        if 'daño' not in df.columns:
            return "Error: El archivo no tiene una columna llamada 'texto'.", None
        
        # Instanciar el modelo
        model = HuggingFaceModel()
        
        # Aplicar predicciones en la columna 'texto'
        df['Predicción'] = df['daño'].apply(model.predict)
        
        # Guardar el archivo con resultados
        output_file = "resultado_predicciones.csv"
        df.to_csv(output_file, index=False, encoding='utf-8')  # Exportar en UTF-8
        
        return "Archivo procesado exitosamente.", output_file
    except Exception as e:
        return f"Error al procesar el archivo: {e}", None


# Interfaz Gradio
def main():
    model = HuggingFaceModel()  # Instancia del modelo para ambas funciones
    
    with gr.Blocks() as demo:
        gr.Markdown("# Modelo Hugging Face - Predicción desde archivo CSV o Texto")
        
        # Sección de predicción desde texto
        gr.Markdown("## Predicción desde texto")
        text_input = gr.Textbox(label="Escribe el texto para predecir")
        text_output = gr.Textbox(label="Predicción", interactive=False)
        predict_btn = gr.Button("Predecir texto")

        # Sección de predicción desde archivo
        gr.Markdown("## Predicción desde archivo CSV")
        file_input = gr.File(label="Sube un archivo .csv", file_types=[".csv"])
        output_message = gr.Textbox(label="Estado del proceso", interactive=False)
        result_file = gr.File(label="Archivo con predicciones")
        submit_btn = gr.Button("Procesar archivo")
        
        # Función de predicción desde archivo
        def handle_process(file):
            message, result_path = process_file(file)
            return message, result_path

        submit_btn.click(
            fn=handle_process,
            inputs=[file_input],
            outputs=[output_message, result_file],
        )

        
        # Función de predicción desde texto
        def predict_text(text):
            return model.predict(text)

        predict_btn.click(
            fn=predict_text,
            inputs=[text_input],
            outputs=[text_output],
        )
    
    demo.launch()

if __name__ == "__main__":
    main()
