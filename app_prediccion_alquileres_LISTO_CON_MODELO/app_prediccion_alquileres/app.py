"""Interfaz Streamlit para el pipeline entrenado de Gradient Boosting.
No sustituye ni reentrena el modelo del informe.
"""
from pathlib import Path
import math
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Estimador de alquileres", page_icon="🏠", layout="centered")
MODEL_PATH = Path(__file__).resolve().parent / "modelo" / "pipeline_gradient_boosting.joblib"
FEATURES = [
    "surface_total_in_m2",
    "surface_covered_in_m2",
    "lat",
    "lon",
    "Num_Banos",
    "Num_Ambientes",
    "Tipo_Propiedad_Limpia",
    "state_name",
    "place_name",
]

@st.cache_resource(show_spinner=False)
def load_model(path: str):
    return joblib.load(path)

st.title("🏠 Predicción de precios de alquiler")
st.write("Estimación experimental de alquileres en **Argentina** mediante el pipeline de Gradient Boosting seleccionado en el informe del curso de Aprendizaje Estadístico.")
st.caption("El resultado es una referencia estadística, no una tasación profesional ni una oferta de alquiler.")

if not MODEL_PATH.exists():
    st.warning("Aún no está instalado el modelo entrenado. Exporta el **pipeline completo** desde Google Colab con el nombre `pipeline_gradient_boosting.joblib` y colócalo dentro de la carpeta `modelo/`.")
    st.info("El formulario se puede revisar, pero no se generarán predicciones ficticias.")

with st.form("vivienda_form"):
    st.subheader("Características de la vivienda")
    col1, col2 = st.columns(2)
    with col1:
        surface_total = st.number_input("Superficie total (m²)", min_value=1.0, value=80.0, step=1.0)
        surface_covered = st.number_input("Superficie cubierta (m²)", min_value=0.0, value=0.0, step=1.0,
                                          help="Introduce 0 solamente si la superficie cubierta real es 0; si es desconocida, indícalo abajo.")
        covered_missing = st.checkbox("No conozco la superficie cubierta", value=True)
        num_banos = st.number_input("Cantidad de baños", min_value=1, max_value=10, value=2)
        num_ambientes = st.number_input("Cantidad de ambientes", min_value=1, max_value=15, value=3)
    with col2:
        tipo = st.selectbox("Tipo de propiedad", ["DEPARTAMENTO", "CASA", "PH", "OTRO"], index=0,
                            help="Debe coincidir con las categorías usadas durante el entrenamiento.")
        provincia = st.text_input("Provincia / estado (`state_name`)", placeholder="Tal como aparece en el dataset")
        localidad = st.text_input("Localidad (`place_name`)", placeholder="Tal como aparece en el dataset")
        lat = st.text_input("Latitud", placeholder="Ej. -34.6")
        lon = st.text_input("Longitud", placeholder="Ej. -58.4")
    st.caption("Se necesitan las 9 variables del informe. Los datos de localización deben corresponder a una propiedad del mercado argentino; no se asignan valores inventados.")
    submitted = st.form_submit_button("Calcular alquiler estimado", type="primary", use_container_width=True)

if submitted:
    problems = []
    if not provincia.strip(): problems.append("Indica la provincia / estado.")
    if not localidad.strip(): problems.append("Indica la localidad.")
    try:
        lat_value = float(lat.strip())
        lon_value = float(lon.strip())
        if not (math.isfinite(lat_value) and -90 <= lat_value <= 90):
            problems.append("La latitud debe estar entre -90 y 90.")
        if not (math.isfinite(lon_value) and -180 <= lon_value <= 180):
            problems.append("La longitud debe estar entre -180 y 180.")
    except ValueError:
        problems.append("Introduce coordenadas numéricas válidas (latitud y longitud).")
        lat_value = lon_value = float("nan")
    if not covered_missing and surface_covered > surface_total:
        problems.append("La superficie cubierta no debería superar la total; comprueba los datos.")

    if problems:
        for problem in problems: st.error(problem)
    elif not MODEL_PATH.exists():
        st.error("No es posible predecir: falta el modelo entrenado exportado desde Colab.")
    else:
        row = {
            "surface_total_in_m2": float(surface_total),
            "surface_covered_in_m2": float("nan") if covered_missing else float(surface_covered),
            "lat": lat_value, "lon": lon_value,
            "Num_Banos": int(num_banos), "Num_Ambientes": int(num_ambientes),
            "Tipo_Propiedad_Limpia": tipo,
            "state_name": provincia.strip(), "place_name": localidad.strip()
        }
        X = pd.DataFrame([row], columns=FEATURES)
        try:
            model = load_model(str(MODEL_PATH))
            if not hasattr(model, "predict"):
                raise TypeError("El archivo no contiene un pipeline predictivo válido.")
            if hasattr(model, "feature_names_in_") and list(model.feature_names_in_) != FEATURES:
                raise ValueError("El orden de variables del modelo no coincide con el informe. Revisa el pipeline exportado.")
            prediction = float(model.predict(X)[0])
            if not math.isfinite(prediction):
                raise ValueError("El modelo devolvió un resultado no finito.")
            st.success("Predicción generada usando el modelo cargado")
            st.metric("Precio mensual estimado", f"US$ {prediction:,.2f}")
            if prediction < 0:
                st.warning("El modelo generó un precio negativo, que no es válido para un alquiler. Revisa datos y modelo.")
            with st.expander("Ver datos enviados al modelo"):
                st.dataframe(X, use_container_width=True, hide_index=True)
            st.caption("Resultados orientativos según el conjunto de entrenamiento argentino. El informe indica errores significativos, especialmente en propiedades de alto precio.")
        except Exception as exc:
            st.error("No se pudo realizar la predicción con el archivo de modelo proporcionado.")
            st.code(str(exc), language="text")
            st.info("Revisa que hayas exportado un pipeline que incluya exactamente el mismo preprocesamiento y modelo entrenado de Colab.")

st.divider()
st.caption("Proyecto académico — Aprendizaje Estadístico | UPAO | No se almacenan los datos ingresados desde esta interfaz.")
