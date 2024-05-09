import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import softmax
import torch
import os  

# Verificar si el archivo comentarios_extraidos.xlsx existe antes de cargarlo
if os.path.exists('comentarios_extraidos.xlsx'):
    # Cargar los comentarios del archivo Excel
    df = pd.read_excel('comentarios_extraidos.xlsx')

    # Eliminar comentarios duplicados
    df.drop_duplicates(subset=['Comentarios'], inplace=True)

    # Cargar el modelo pre-entrenado de BERT para análisis de sentimientos
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Configurar el dispositivo para ejecutar el modelo (CPU o GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Función para analizar el sentimiento de un comentario utilizando BERT
    def analizar_sentimiento(comentario):
        inputs = tokenizer(comentario, return_tensors="pt", max_length=512, truncation=True)
        inputs.to(device)
        outputs = model(**inputs)
        probabilidades = softmax(outputs.logits, dim=1)
        return probabilidades[0][1].item()  # Suponiendo que estamos interesados en la probabilidad de la clase positiva

    # Función para clasificar el sentimiento en palabras
    def clasificar_sentimiento(sentimiento):
        if sentimiento >= 0.7:
            return 'Positivo'
        elif sentimiento <= 0.5:
            return 'Negativo'
        else:
            return 'Neutro'

    # Analizar el sentimiento de cada comentario y agregar los resultados al DataFrame
    df['Sentimiento'] = df['Comentarios'].apply(analizar_sentimiento)

    # Clasificar los sentimientos en palabras
    df['Sentimiento_palabra'] = df['Sentimiento'].apply(clasificar_sentimiento)
    
    # Eliminar la columna 'Sentimiento' del DataFrame
    df.drop(columns=['Sentimiento'], inplace=True)

    # Guardar el DataFrame con los resultados del análisis de sentimiento en un nuevo archivo Excel
    df.to_excel('comentarios_analizados.xlsx', index=False)

    print("Comentarios analizados con BERT y clasificados por palabras guardados en comentarios_analizados.xlsx")
else:
    print("No se encontró el archivo comentarios_extraidos.xlsx. Asegúrate de haber extraído comentarios antes de analizarlos.")
