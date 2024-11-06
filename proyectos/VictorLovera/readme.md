# Análisis de Sentimientos con Lógica Difusa

Este proyecto es mi implementacion del algoritmo de análisis de sentimiento difuso a partir de tweets sugerido por [Vashishtha, S., & Susan, S. (2019). Fuzzy rule based unsupervised sentiment analysis from social media posts](https://www.researchgate.net/profile/Srishti-Vashishtha-2/publication/334622166_Fuzzy_Rule_based_Unsupervised_Sentiment_Analysis_from_Social_Media_Posts/links/5ece42174585152945149e5b/Fuzzy-Rule-based-Unsupervised-Sentiment-Analysis-from-Social-Media-Posts.pdf).

Este tiene como objetivo realizar un análisis de sentimientos sobre un conjunto de oraciones utilizando una combinación de análisis de sentimiento basado en un léxico y un sistema de lógica difusa para inferir la polaridad (positiva, negativa o neutral). 

### Características principales:
- Preprocesamiento del texto (limpieza de URLs, menciones, números y caracteres especiales).
- Análisis de sentimientos con el **SentimentIntensityAnalyzer** de **NLTK**.
- Clasificación de sentimientos utilizando lógica difusa, con base en los puntajes de sentimientos positivo y negativo.
- Cálculo de un puntaje final de inferencia utilizando reglas difusas para categorizar las oraciones como positivas, negativas o neutrales.
- Cálculo del tiempo de ejecución para cada oración procesada, así como del tiempo promedio total.
- Generación de un archivo CSV con los resultados detallados y un reporte sobre el tiempo de ejecución.

## Requerimientos

Este proyecto requiere las siguientes librerías de Python:

- `numpy`
- `pandas`
- `skfuzzy`
- `nltk`

Puedes instalar las dependencias necesarias ejecutando el siguiente comando:

```bash
pip install -r requirements.txt
```

## Estructura de archivos

- `Proyecto.py`: Archivo principal que contiene el código para realizar el análisis de sentimientos, el preprocesamiento y la lógica difusa. 
- `Proyecto.ipynb`: Notebook que contiene el codigo fuente ademas de una explicacion del funcionamiento de cada componente del mismo.
- `Data/`: Carpeta que contiene los archivos de datos (CSV) para el análisis.
- `requirements.txt`: Archivo con las dependencias necesarias para ejecutar el proyecto.

## Uso

1. Coloca tus archivos CSV con oraciones en la carpeta `Data/`.
2. Ejecuta el archivo `Proyecto.py`:

```bash
py Proyecto.py
```

3. El programa te pedirá que selecciones el archivo CSV con los datos a procesar.
4. Los resultados será mostrado en la consola y se guardarán en un nuevo archivo CSV con la siguiente información:
    - Oración original
    - Etiqueta original
    - Puntaje positivo
    - Puntaje negativo
    - Puntaje final
    - Tiempo de ejecución

5. El tiempo promedio total de ejecución será mostrado en la consola y se guardarán en un nuevo archivo CSV.

## Ejemplo de salida:
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Oración</th>
      <th>Etiqueta original</th>
      <th>Puntaje Positivo</th>
      <th>Puntaje Negativo</th>
      <th>Resultado de inferencia</th>
      <th>Tiempo Computo</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>i loooooooovvvvvveee my kindle not that the dx...</td>
      <td>1</td>
      <td>0.382</td>
      <td>0.000</td>
      <td>5.371795</td>
      <td>0.002509</td>
    </tr>
    <tr>
      <th>1</th>
      <td>reading my kindle love it lee childs is good read</td>
      <td>1</td>
      <td>0.470</td>
      <td>0.000</td>
      <td>5.963203</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ok first assesment of the kindle it fucking rocks</td>
      <td>1</td>
      <td>0.216</td>
      <td>0.000</td>
      <td>4.890017</td>
      <td>0.000999</td>
    </tr>
    <tr>
      <th>3</th>
      <td>you ll love your kindle i ve had mine for a fe...</td>
      <td>1</td>
      <td>0.204</td>
      <td>0.135</td>
      <td>4.685033</td>
      <td>0.001006</td>
    </tr>
    <tr>
      <th>4</th>
      <td>fair enough but i have the kindle and i think ...</td>
      <td>1</td>
      <td>0.456</td>
      <td>0.000</td>
      <td>5.849136</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>
