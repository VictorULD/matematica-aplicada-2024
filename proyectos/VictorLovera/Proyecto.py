import os
import numpy as np
import pandas as pd
import re
import skfuzzy as fuzz
import time
from nltk.sentiment import SentimentIntensityAnalyzer

# Función para leer un archivo CSV y añadir una columna de índice 'Id'
def Leer_Archivo(Archivo):
    data = pd.read_csv(Archivo)  # Carga los datos desde el archivo CSV
    data.insert(0, 'Id', range(1, 1 + len(data)))  # Añade una columna 'Id' como índice
    return data

# Función para limpiar y preprocesar texto
def Pre_proceso_Texto(data):
    def process(text):
        # Eliminar enlaces, menciones y caracteres especiales del texto
        text = re.sub(r'http[s]?://\S+', '', text)
        text = re.sub(r' www\S+', '', text)
        text = re.sub(r'@\S+', '', text)
        text = re.sub(r'[^\w\s]|[\d]', ' ', text)
        text = re.sub(r'\s\s+', ' ', text)
        text = re.sub(r"#", "", text)

        # Expandir contracciones comunes en el texto
        text = re.sub(r"n\'t", " not", text)
        text = re.sub(r"\'re", " are", text)
        text = re.sub(r"\'s", " is", text)
        text = re.sub(r"\'d", " would", text)
        text = re.sub(r"\'ll", " will", text)
        text = re.sub(r"\'t", " not", text)
        text = re.sub(r"\'ve", " have", text)
        text = re.sub(r"\'m", " am", text)

        # Eliminar caracteres no ASCII y convertir el texto a minúsculas
        text = text.strip().lower().encode('ascii', 'ignore').decode()
        
        return text

    # Procesar cada fila en la columna 'sentence' aplicando la limpieza de texto
    for i, row in data.iterrows():
        row['sentence'] = process(row['sentence'])
        
    return data

# Función para realizar análisis de sentimientos con la librería nltk
def Lexicon_Sentiminetos(data):
    sia = SentimentIntensityAnalyzer()  # Inicializa el analizador de sentimientos
    res = {}
    tiempos = []  # Lista para almacenar tiempos de ejecución por cada análisis

    # Realiza análisis de sentimientos en cada fila
    for i, row in data.iterrows():
        start_time = time.time()  # Registra el tiempo de inicio
        text = row['sentence']
        myid = row['Id']
        res[myid] = sia.polarity_scores(text)  # Guarda el análisis de sentimientos
        
        tiempos.append(time.time() - start_time)  # Calcula el tiempo de ejecución

    # Convierte los resultados en un DataFrame
    Datos_Sent = pd.DataFrame(res).T
    Datos_Sent = Datos_Sent.reset_index().rename(columns={'index': 'Id'})
    Datos_Sent['Tiempo Computo'] = tiempos  # Agrega la columna de tiempo de ejecución
    
    # Combina los resultados con el DataFrame original y elimina columnas innecesarias
    Datos_Sent = Datos_Sent.merge(data, how='left').drop(['compound', 'neu'], axis=1)
    
    return Datos_Sent

# Función para contar oraciones positivas y negativas basadas en el resultado de inferencia
def Total_Pos_Neg(data):
    pos_t = 0  # Contador de oraciones positivas
    neg_t = 0  # Contador de oraciones negativas
    
    # Cuenta cada oración según su clasificación en 'Resultado de inferencia'
    for i, row in data.iterrows():
        if row['Resultado de inferencia'] < 5:
            neg_t += 1
        else:
            pos_t += 1
        
    return pos_t, neg_t

# Función para ejecutar el procesamiento de datos, incluyendo lógica difusa
def Computar_Datos(Archivo):
    # Define los rangos para los conjuntos difusos de los sentimientos y el resultado
    Sent_Postivo = np.arange(0, 1, 0.01)
    Sent_Negativo = np.arange(0, 1, 0.01)
    Out_U = np.arange(0, 10, 1)

    # Define las funciones de membresía para los conjuntos difusos
    sp_lo = fuzz.trimf(Sent_Postivo, [0, 0, 0.5])
    sp_med = fuzz.trimf(Sent_Postivo, [0, 0.5, 1])
    sp_hi = fuzz.trimf(Sent_Postivo, [0.5, 1, 1])

    sn_lo = fuzz.trimf(Sent_Negativo, [0, 0, 0.5])
    sn_med = fuzz.trimf(Sent_Negativo, [0, 0.5, 1])
    sn_hi = fuzz.trimf(Sent_Negativo, [0.5, 1, 1]) 

    out_neg = fuzz.trimf(Out_U, [0, 0, 5])
    out_neu = fuzz.trimf(Out_U, [0, 5, 10])
    out_pos = fuzz.trimf(Out_U, [5, 5, 10])

    # Función interna para calcular la inferencia difusa
    def Fuzzy_DeFuzzy(scoreposi, scoreneg):
        # Calcula el grado de membresía para cada regla difusa
        pos_n_lo = fuzz.interp_membership(Sent_Postivo, sp_lo, scoreposi)
        pos_n_med = fuzz.interp_membership(Sent_Postivo, sp_med, scoreposi)
        pos_n_hi = fuzz.interp_membership(Sent_Postivo, sp_hi, scoreposi)

        neg_n_lo = fuzz.interp_membership(Sent_Negativo, sn_lo, scoreneg)
        neg_n_med = fuzz.interp_membership(Sent_Negativo, sn_med, scoreneg)
        neg_n_hi = fuzz.interp_membership(Sent_Negativo, sn_hi, scoreneg)

        # Aplica las reglas difusas y calcula la inferencia
        regla1 = np.fmin(pos_n_lo, neg_n_lo)
        regla2 = np.fmin(pos_n_med, neg_n_lo) 
        regla3 = np.fmin(pos_n_hi, neg_n_lo) 
        regla4 = np.fmin(pos_n_lo, neg_n_med) 
        regla5 = np.fmin(pos_n_med, neg_n_med) 
        regla6 = np.fmin(pos_n_hi, neg_n_med) 
        regla7 = np.fmin(pos_n_lo, neg_n_hi) 
        regla8 = np.fmin(pos_n_med, neg_n_hi) 
        regla9 = np.fmin(pos_n_hi, neg_n_hi)

        # Define el resultado agregado de cada regla
        temp1 = np.fmax(regla4, regla7)
        regla_neg = np.fmax(temp1, regla8)
        out_act_low = np.fmin(regla_neg, out_neg)

        temp2 = np.fmax(regla1, regla5)
        regla_neu = np.fmax(temp2, regla9)
        out_act_neu = np.fmin(regla_neu, out_neu)

        temp3 = np.fmax(regla2, regla3)
        regla_pos = np.fmax(temp3, regla6)
        out_act_pos = np.fmin(regla_pos, out_pos)

        # Devuelve el valor de defuzzificación
        aggregated = np.fmax(out_act_low, np.fmax(out_act_neu, out_act_pos))
        return fuzz.defuzz(Out_U, aggregated, 'centroid')

    # Llama a funciones de procesamiento y análisis
    Datos = Leer_Archivo(Archivo)
    Datos = Pre_proceso_Texto(Datos)
    Datos = Lexicon_Sentiminetos(Datos)
    
    Resultados = Datos.copy()

    # Calcula el resultado difuso para cada fila y registra el tiempo
    tiempos = []
    for i, row in Datos.iterrows():
        start_time = time.time()
        Resultados.loc[i, 'Results'] = Fuzzy_DeFuzzy(row['pos'], row['neg'])
        tiempos.append(time.time() - start_time)

    Resultados['Tiempo Computo'] += tiempos  # Suma los tiempos al tiempo total
    
    # Ajuste final del DataFrame de resultados
    Resultados.drop('Id', axis=1)
    Resultados = Resultados[['sentence', 'sentiment', 'pos', 'neg', 'Results', 'Tiempo Computo']]
    Resultados.columns = ['Oración', 'Etiqueta original', 'Puntaje Positivo', 'Puntaje Negativo', 'Resultado de inferencia', 'Tiempo Computo']

    return Resultados

def seleccionar_archivo(carpeta):
    # Obtener la lista de archivos en la carpeta especificada
    archivos = [f for f in os.listdir(carpeta) if os.path.isfile(os.path.join(carpeta, f))]
    
    if not archivos:
        print("No se encontraron archivos en la carpeta.")
        return None

    # Mostrar los archivos disponibles
    print("Archivos disponibles:")
    for i, archivo in enumerate(archivos, start=1):
        print(f"{i}. {archivo}")
    
    # Pedir al usuario que seleccione un archivo
    try:
        seleccion = int(input("Elige el número del archivo que deseas usar: "))
        if 1 <= seleccion <= len(archivos):
            archivo_seleccionado = archivos[seleccion - 1]
            print(f"Has seleccionado: {archivo_seleccionado}")
            return archivo_seleccionado
        else:
            print("Selección no válida. Intenta de nuevo.")
            return None
    except ValueError:
        print("Por favor, ingresa un número válido.")
        return None

# Función principal para ejecutar el programa
def main():
    carpeta = './Data'  # Especifica la carpeta donde se encuentran los archivos
    archivo = os.path.join(carpeta, seleccionar_archivo(carpeta))
    Resultados = Computar_Datos(archivo)

    print(Resultados)
    print("\nEl Promedio temporal de Ejecucion es: ", Resultados['Tiempo Computo'].mean())
    
    Total_Positivos, Total_Negativos = Total_Pos_Neg(Resultados)
    print("\nLa cantidad de Oraciones Positivas es: ", Total_Positivos)
    print("\nLa cantidad de Oraciones Negativas es: ", Total_Negativos)
    
    # Guardar el archivo CSV
    output_filename = 'resultado_sentimiento.csv'
    Resultados.to_csv(output_filename, index=False)
    
    
    total_oraciones = len(Resultados)
    tiempo_promedio = Resultados['Tiempo Computo'].mean()
    resumen_df = pd.DataFrame({
        'Total de Oraciones': [total_oraciones],
        'Tiempo Promedio (segundos)': [tiempo_promedio]
    })
    
    resumen_df.to_csv('Reporte Tiempo de ejecucion.csv', index=False)
    
# Ejecuta el programa si se llama directamente
if __name__ == "__main__":
    main()

