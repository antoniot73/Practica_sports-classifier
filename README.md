# Practica_sports-classifier
Implementación de una CNN para clasificación multiclase de imágenes deportivas.

# Clasificador de Imágenes Deportivas con CNN en Python

Proyecto de percepción computacional **alojado en GitHub** y **ejecutado en Google Colab** para entrenar, evaluar y utilizar una **red neuronal convolucional (CNN)** orientada a la clasificación de imágenes deportivas. El sistema distingue dos clases: **golf** y **basket**, a partir de un pipeline que integra carga del dataset, preprocesamiento, entrenamiento, evaluación, análisis de errores e inferencia sobre imágenes nuevas.  

## Descripción general

Este proyecto implementa un flujo de trabajo supervisado para clasificación de imágenes RGB de **21 × 28 píxeles**. El código fue desarrollado como práctica académica de **Percepción Computacional**, se encuentra versionado en **GitHub** y fue ejecutado en **Google Colab**, apoyándose en **Google Drive** para acceder al dataset y guardar el modelo entrenado.  

## Flujo general del sistema

1. Carga del notebook/script desde GitHub en Google Colab.
2. Montaje de Google Drive.
3. Lectura recursiva de imágenes del dataset.
4. Generación de etiquetas por clase.
5. Conversión de imágenes a arreglos de NumPy.
6. División en entrenamiento, validación y prueba.
7. Normalización de datos y codificación *one-hot*.
8. Definición de la arquitectura CNN.
9. Entrenamiento del modelo.
10. Evaluación cuantitativa sobre datos no vistos.
11. Visualización de curvas de entrenamiento.
12. Análisis de aciertos y errores.
13. Predicción de imágenes nuevas.
14. Guardado del modelo entrenado en formato `.keras`.  

## Entorno de trabajo

| Elemento                       | Especificación                                     |
| ------------------------------ | -------------------------------------------------- |
| Repositorio                    | GitHub                                             |
| Entorno de ejecución           | Google Colab                                       |
| Lenguaje                       | Python 3.12.13                                     |
| Sistema operativo del entorno  | Linux 6.6.113+-x86_64-with-glibc2.35               |
| Bibliotecas principales        | NumPy, Matplotlib, scikit-learn, TensorFlow, Keras |
| Almacenamiento de datos/modelo | Google Drive                                       |
| Aceleración                    | GPU en Colab                                       |


## Dataset

El pipeline trabaja con **11,623 imágenes** distribuidas en dos clases:

* **golf**
* **basket**

Las imágenes se leen desde directorios en Google Drive mediante `os.walk()` y se convierten a arreglos de NumPy para su procesamiento. El reporte también evidencia un desbalance entre clases, factor que influye en el desempeño del modelo.  

## Objetivo

Desarrollar un sistema reproducible de clasificación de imágenes deportivas mediante aprendizaje profundo, capaz de reconocer patrones visuales y evaluar críticamente su comportamiento con métricas, curvas de entrenamiento y análisis de errores. 

## Arquitectura del modelo

La red neuronal convolucional implementada sigue esta estructura:

* `Input(shape=(21, 28, 3))`
* `Conv2D(32, kernel_size=(3,3), padding='same', activation='linear')`
* `LeakyReLU(negative_slope=0.1)`
* `MaxPooling2D((2,2), padding='same')`
* `Dropout(0.5)`
* `Flatten()`
* `Dense(32, activation='linear')`
* `LeakyReLU(negative_slope=0.1)`
* `Dropout(0.5)`
* `Dense(nClasses, activation='softmax')`

El modelo cuenta con **158,690 parámetros entrenables**.  

## Configuración de entrenamiento

* Optimizador: **Adagrad**
* `learning_rate = 1e-3`
* `weight_decay = learning_rate / 100`
* Épocas: **6**
* `batch_size = 64`
* Función de pérdida: **categorical crossentropy**

Antes del entrenamiento, las imágenes se convierten a `float32`, se normalizan al rango `[0,1]` y las etiquetas se transforman con `to_categorical()`.  

## Resultados principales

El modelo obtuvo:

* **Test accuracy:** `0.8615`
* **Test loss:** `0.2735`
* **2003 aciertos**
* **322 errores**

El análisis por clase mostró un comportamiento desigual:

* **golf:** recall `1.00`
* **basket:** recall `0.16`

Esto indica que la exactitud global es buena, pero la generalización entre clases no es equilibrada. 

## Predicción de nuevas imágenes

El pipeline permite inferencia sobre imágenes externas redimensionadas a **21 × 28**, usando `resize()` y `predict()`. 

## Estructura sugerida del proyecto

| Archivo                                       | Descripción                                     |
| --------------------------------------------- | ----------------------------------------------- |
| `CNN_Clasificador_Deportes.ipynb`             | Notebook principal ejecutado en Google Colab    |
| `Reporte_Colab_CNN_Tarea03_Antonio_Toro.pdf`  | Reporte de resultados de pesos del modelo       |
| `README.md`                                   | Documentación general del repositorio           |

## Dependencias

```bash
pip install numpy matplotlib scikit-learn tensorflow keras scikit-image
```

## Ejecución en Google Colab

1. Abrir el notebook del repositorio GitHub en Google Colab.
2. Montar Google Drive.
3. Ajustar las rutas del dataset y de salida del modelo.
4. Ejecutar todas las celdas del notebook.
5. Evaluar resultados y realizar predicciones con nuevas imágenes. 

## Limitaciones

* Dependencia de rutas específicas en Google Drive.
* Dataset desbalanceado.
* Arquitectura CNN simple.
* Desempeño desigual entre clases.  

## Mejoras futuras

* Balanceo del dataset.
* Aumento de datos (*data augmentation*).
* Arquitecturas más profundas.
* Ajuste de hiperparámetros.
* Portabilidad para ejecución fuera de Colab. 

## Usuario del script 

**Antonio Nicolás Toro González**
Práctica desarrollada para la asignatura **Percepción Computacional** de la **Maestría en Inteligencia Artificial para la Transformación Digital**. 

## Licencia

Material compartido con fines académicos y educativos. 

