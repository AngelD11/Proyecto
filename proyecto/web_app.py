from flask import Flask, render_template, request, redirect, url_for
from app import extraer_comentarios
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import softmax
import torch
import os
import analisis
from analisis import clasificar_sentimiento


app = Flask(__name__)

# Función para analizar el sentimiento de un comentario utilizando BERT
def analizar_sentimiento(comentario):
    # Cargar el modelo pre-entrenado de BERT para análisis de sentimientos
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Configurar el dispositivo para ejecutar el modelo (CPU o GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    inputs = tokenizer(comentario, return_tensors="pt", max_length=512, truncation=True)
    inputs.to(device)
    outputs = model(**inputs)
    probabilidades = softmax(outputs.logits, dim=1)
    return probabilidades[0][1].item()  # Suponiendo que estamos interesados en la probabilidad de la clase positiva

# Ruta para la página de inicio
@app.route('/')
def index():
    mensaje = request.args.get('mensaje')
    return render_template('index.html', mensaje=mensaje)

# Ruta para la página de todos los productos
@app.route('/todos_productos')
def todos_productos():
    if os.path.exists('comentarios_analizados.xlsx'):
        try:
            # Cargar los datos del archivo comentarios_extraidos.xlsx en un DataFrame
            df = pd.read_excel('comentarios_analizados.xlsx')
            # Convertir el DataFrame a un diccionario para pasar al contexto de la plantilla
            datos = df.to_dict(orient='records')
            # Renderizar la plantilla HTML con los datos del DataFrame
            return render_template('todos_productos.html', datos=datos)
        except FileNotFoundError:
            return render_template('todos_productos.html', mensaje='No hay comentarios disponibles')
    else:
        return render_template('todos_productos.html', mensaje='No hay comentarios disponibles')

# Ruta para la página de estadísticas
@app.route('/estadisticas')
def estadisticas():
    # Aquí podrías calcular y pasar los datos de las estadísticas como contexto
    return render_template('estadisticas.html')

# Ruta para la página de los mejores calificados
@app.route('/mejores_calificados')
def mejores_calificados():
    # Aquí podrías obtener los datos de los productos mejor calificados y pasarlos como contexto
    return render_template('mejores_calificados.html')

# Ruta para extraer comentarios
@app.route('/extraer_comentarios', methods=['POST'])
def extraer():
    url = request.form['url']
    clase = 'a-row a-spacing-small review-data'
    extraer_comentarios(url, clase)
    return redirect(url_for('index', mensaje='Comentarios extraídos con éxito'))

# Ruta para analizar comentarios
@app.route('/analizar_comentarios', methods=['POST'])
def analizar():
    # Llama a la función para analizar los comentarios
    # Cargar los comentarios del archivo Excel
    df = pd.read_excel('comentarios_extraidos.xlsx')

    # Eliminar comentarios duplicados
    df.drop_duplicates(subset=['Comentarios'], inplace=True)

    # Analizar el sentimiento de cada comentario y agregar los resultados al DataFrame
    df['Sentimiento'] = df['Comentarios'].apply(analizar_sentimiento)

    # Clasificar los sentimientos en palabras
    df['Sentimiento_palabra'] = df['Sentimiento'].apply(clasificar_sentimiento)

    # Guardar el DataFrame con los resultados del análisis de sentimiento en un nuevo archivo Excel
    df.to_excel('comentarios_analizados.xlsx', index=False)

    print("Comentarios analizados con BERT y clasificados por palabras guardados en comentarios_analizados.xlsx")

    return redirect(url_for('index', mensaje='Comentarios analizados con éxito'))

if __name__ == '__main__':
    app.run(debug=True)
