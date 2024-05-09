import requests
from bs4 import BeautifulSoup
import pandas as pd

# Función para extraer comentarios de una URL y guardarlos en un archivo Excel
def extraer_comentarios(url, clase):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Lanza una excepción si la solicitud no es exitosa (código de estado diferente de 200)
        soup = BeautifulSoup(response.text, 'html.parser')
        elementos = soup.find_all(class_=clase)
        comentarios_nuevos = [elemento.text.strip() for elemento in elementos]
        
        if not comentarios_nuevos:
            print("No se encontraron comentarios en la página.")
            return
        
        print("Comentarios encontrados:", comentarios_nuevos)
        
        try:
            df_existente = pd.read_excel('comentarios_extraidos.xlsx')
            print("DataFrame existente:", df_existente)
        except FileNotFoundError:
            df_existente = pd.DataFrame()
        
        # Filtrar comentarios nuevos que no estén en el DataFrame existente
        comentarios_filtrados = []
        for comentario in comentarios_nuevos:
            comentario_sin_ultima_palabra = comentario.rsplit(' ', 1)[0]
            comentario_sin_palabra_especifica = comentario_sin_ultima_palabra.replace("Leer", "")
            comentarios_filtrados.append(comentario_sin_palabra_especifica)

        print("Comentarios filtrados:", comentarios_filtrados)
        
        if comentarios_filtrados:
            df_nuevo = pd.DataFrame({'Comentarios': comentarios_filtrados})
            
            # Si el archivo ya existe, concatena los nuevos comentarios
            if not df_existente.empty:
                df_final = pd.concat([df_existente, df_nuevo], ignore_index=True)
            else:
                df_final = df_nuevo
            
            # Guardar en el archivo comentarios_extraidos.xlsx
            df_final.to_excel('comentarios_extraidos.xlsx', index=False)
            print("Comentarios agregados al archivo comentarios_extraidos.xlsx")
        else:
            print("No se encontraron comentarios nuevos para agregar.")
    except Exception as e:
        print(f"Error al extraer comentarios: {e}")
