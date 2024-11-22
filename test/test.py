import unittest
from unittest.mock import patch, MagicMock
from src.main import HuggingFaceModel, predict_individual_text, process_file

class TestHuggingFaceModel(unittest.TestCase):
    def setUp(self):
        """Configuración inicial para las pruebas."""
        self.model = HuggingFaceModel()

    def test_predict_output(self):
        """Prueba que la salida de predicción sea una etiqueta válida."""
        text = "un árbol cortó las cuerdas"
        prediction = self.model.predict(text)
        
        # Verificar que la predicción sea un string
        self.assertIsInstance(prediction, str)
        
        # Verificar que la predicción esté en las etiquetas válidas
        self.assertIn(prediction, ["ejecutar", "cancelar"])

    def test_validate_input(self):
        """Prueba que la función de validación maneje correctamente las entradas."""
        
        # Casos válidos
        valid_texts = [
            "Este es un texto válido.",
            "1234 es un número en el texto.",
            "Texto con símbolos como !, ?, ."
        ]
        for text in valid_texts:
            self.assertTrue(self.model.validate_input(text))
        
        # Casos inválidos
        invalid_texts = [
            "  ",  # Solo espacios
            "123",  # Solo números
            "Texto con caracteres inválidos #$%&"  # Caracteres no permitidos
        ]
        for text in invalid_texts:
            self.assertFalse(self.model.validate_input(text))

class TestGradioInterface(unittest.TestCase):
    def test_predict_individual_text(self):
        """Prueba que la función predict_individual_text devuelva la salida correcta."""
        input_text = "no hay energía en el sector"
        prediction = predict_individual_text(input_text)
        
        # Verificar que la predicción sea un string
        self.assertIsInstance(prediction, str)
        
        # Verificar que la predicción esté en las etiquetas válidas
        self.assertIn(prediction, ["ejecutar", "cancelar"])

class TestProcessFile(unittest.TestCase):

    @patch('pandas.read_csv')  # Mock de la función read_csv de pandas
    def test_process_file_success(self, mock_read_csv):
        """Prueba para procesar correctamente el archivo CSV y generar las predicciones."""
        # Crear un DataFrame simulado
        mock_df = MagicMock()
        mock_df.columns = ['Descripción de la orden']
        mock_df['Descripción de la orden'] = ["Texto válido", "Otro texto"]
        
        mock_read_csv.return_value = mock_df  # Simular que read_csv devuelve este DataFrame
        
        # Mock para la función HuggingFaceModel.predict
        with patch.object(HuggingFaceModel, 'predict', return_value="ejecutar") as mock_predict:
            message, output_file = process_file(MagicMock(name='file'))
            
            # Verificar que el mensaje de éxito se retorne
            self.assertEqual(message, "Archivo procesado exitosamente.")
            self.assertIsNotNone(output_file)  # Verificar que el archivo de salida no es None
            mock_predict.assert_called()

    @patch('pandas.read_csv')  # Mock de la función read_csv de pandas
    def test_process_file_no_text_column(self, mock_read_csv):
        """Prueba que maneje el caso donde el archivo no tiene la columna 'Descripción de la orden'."""
        mock_df = MagicMock()
        mock_df.columns = ['otra_columna']
        
        mock_read_csv.return_value = mock_df
        
        message, output_file = process_file(MagicMock(name='file'))
        
        # Verificar que el mensaje de error se retorne
        self.assertEqual(message, "Error: El archivo no tiene una columna llamada 'Descripción de la orden'.")
        self.assertIsNone(output_file)  # No debería generarse archivo si hay un error

    @patch('pandas.read_csv')  # Mock de la función read_csv de pandas
    def test_process_file_exception(self, mock_read_csv):
        """Prueba para manejar excepciones al procesar el archivo."""
        mock_read_csv.side_effect = Exception("Error en la lectura del archivo")
        
        message, output_file = process_file(MagicMock(name='file'))
        
        # Verificar que el mensaje de error se retorne
        self.assertEqual(message, "Error al procesar el archivo: Error en la lectura del archivo")
        self.assertIsNone(output_file)

if __name__ == "__main__":
    unittest.main()
