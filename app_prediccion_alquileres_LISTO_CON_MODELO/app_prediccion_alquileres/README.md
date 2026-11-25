# Modelo de predicción de precios de alquiler de viviendas

Aplicación web académica con Streamlit para utilizar **el modelo Gradient Boosting ya entrenado** del proyecto de Aprendizaje Estadístico (UPAO). Los datos del informe corresponden a alquileres de Argentina (Properati).

## Estado actual

El archivo `modelo/pipeline_gradient_boosting.joblib` proporcionado por el equipo **ya está incluido** en este proyecto. El notebook contiene `modelo_gb` como pipeline con preprocesamiento y Gradient Boosting. El modelo fue serializado con **scikit-learn 1.6.1**, por lo que `requirements.txt` fija dicha versión para reproducir el entorno. **La predicción completa aún debe verificarse ejecutando la app con ese entorno compatible**; no se ha publicado todavía ninguna web. El ejemplo de 80 m², 2 baños y 3 ambientes no basta por sí solo: hay que indicar ubicación y superficie cubierta (o marcarla desconocida).

## 1. Exportar de nuevo el pipeline desde Google Colab (opcional)

Abre `Modelo_prediccion_alquiler_CON_EXPORTACION.ipynb` en Google Colab. Es una copia del notebook original que incluye al final la celda de exportación. Monta Google Drive y ejecuta las celdas de preparación y entrenamiento (al menos hasta la celda 22, en orden). La celda añadida descarga `pipeline_gradient_boosting.joblib` y comprueba que las predicciones se mantienen al guardar y cargar el pipeline.

Alternativamente, justo después de `modelo_gb.fit(X_train, y_train)` puedes ejecutar:

```python
import joblib
from google.colab import files
joblib.dump(modelo_gb, "/content/pipeline_gradient_boosting.joblib")
files.download("/content/pipeline_gradient_boosting.joblib")
```

**El notebook original depende de** `/content/drive/MyDrive/Aprendizaje_Estadistico/dataset a usar.zip`. Necesitas que esté disponible en esa ubicación o cambiar la ruta para reentrenar. No se incluye este dataset en el ZIP.

El modelo ya viene incluido. Solo necesitarías reemplazarlo si vuelves a entrenar el pipeline en Colab.

## 2. Ejecutar localmente

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

Se abrirá una dirección local como `http://localhost:8501`.

## 3. Variables que espera el modelo

| Campo | Columna usada en el informe |
|---|---|
| Área total | `surface_total_in_m2` |
| Área cubierta | `surface_covered_in_m2` |
| Latitud | `lat` |
| Longitud | `lon` |
| Baños | `Num_Banos` |
| Ambientes | `Num_Ambientes` |
| Tipo de vivienda | `Tipo_Propiedad_Limpia` |
| Provincia/estado | `state_name` |
| Localidad | `place_name` |

La aplicación inicia con **80 m², 2 baños, 3 ambientes y DEPARTAMENTO**, como ejemplo solicitado. Debes indicar provincia, localidad, latitud y longitud; la superficie cubierta puede marcarse como desconocida para que el pipeline la impute. Los nombres/categorías de provincia, localidad y tipo deben corresponder a los utilizados en el entrenamiento. En el informe se indica que las categorías se normalizaron a mayúsculas en algunos campos; **verifica las etiquetas reales de `Tipo_Propiedad_Limpia` y el tratamiento de `state_name`/`place_name` en Colab** para asegurar coincidencia.

## 4. Publicación en GitHub y Streamlit Community Cloud

1. Crea un repositorio de GitHub y sube `app.py`, `requirements.txt`, `README.md` y la carpeta `modelo/` con el archivo `.joblib` si su tamaño y licencia permiten publicarlo.
2. Prueba primero de forma local usando registros cuyo resultado conozcas de Colab.
3. En [Streamlit Community Cloud](https://share.streamlit.io/) inicia sesión con GitHub, selecciona el repositorio, la rama y `app.py`, y despliega la aplicación.
4. Copia el enlace de la app y el enlace del repositorio a la sección VII de tu informe. Añade capturas de la aplicación ejecutándose y de las pruebas.

Evita subir datos privados y revisa la licencia y el tamaño de cualquier artefacto antes de publicarlo. Los modelos de `joblib` son archivos serializados de Python: **carga únicamente archivos producidos por tu equipo o de una fuente confiable**. El modelo usa scikit-learn 1.6.1. Usa Python 3.11 o 3.12 y las dependencias del archivo `requirements.txt` para el despliegue. Si vuelves a exportar con otra versión, actualiza las dependencias.

## 5. Límites del proyecto

El informe reporta `MAE=988.99 USD`, `RMSE=3591.81 USD` y `R²=0.2712` de promedio en validación cruzada para Gradient Boosting. Son resultados experimentales, no un margen de error garantizado para cada vivienda. El sistema se diseñó con datos de Argentina, por lo que **no debe trasladarse directamente al mercado de Perú**.
