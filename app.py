from collections import Counter
from pathlib import Path

import altair as alt
import joblib
import pandas as pd
import streamlit as st


MODEL_PATH = Path(__file__).with_name("modelo_regresion_docentes.pkl")
DATA_PATH = Path(__file__).with_name("respuestas_analisis2.csv")
FEATURES = ["variabilidad", "uniformidad"]
CONVERSION = {"A": 5, "B": 4, "C": 3, "D": 2, "E": 1}
COLUMNAS_CSV = ["question_id", "value_item", "teacher_id", "student_id", "course_id"]


@st.cache_resource
def cargar_modelo():
    return joblib.load(MODEL_PATH)


def interpretar_promedio(promedio):
    if promedio >= 4.5:
        return "Desempeño excelente"
    if promedio >= 4:
        return "Desempeño bueno"
    if promedio >= 3:
        return "Desempeño regular"
    return "Desempeño bajo o en riesgo"


def clasificar_encuesta(promedio, variabilidad, uniformidad):
    if uniformidad > 0.8:
        return "Encuesta sospechosa"
    if promedio > 4.5 and variabilidad <= 2:
        return "Posible sesgo positivo"
    if variabilidad >= 3:
        return "Encuesta confiable"
    return "Encuesta dudosa"


def analizar_respuestas(respuestas):
    promedio = sum(respuestas) / len(respuestas)
    variabilidad = len(set(respuestas))
    conteo = Counter(respuestas)
    uniformidad = conteo.most_common(1)[0][1] / len(respuestas)
    resultado = clasificar_encuesta(promedio, variabilidad, uniformidad)
    return promedio, variabilidad, uniformidad, resultado


def predecir(modelo, variabilidad, uniformidad):
    entrada = pd.DataFrame(
        [[variabilidad, uniformidad]],
        columns=FEATURES,
    )
    return float(modelo.predict(entrada)[0])


def mostrar_grafico_pastel(datos, categoria, valor, titulo):
    grafico = (
        alt.Chart(datos)
        .mark_arc(innerRadius=45)
        .encode(
            theta=alt.Theta(field=valor, type="quantitative"),
            color=alt.Color(field=categoria, type="nominal"),
            tooltip=[
                alt.Tooltip(field=categoria, type="nominal", title=categoria),
                alt.Tooltip(field=valor, type="quantitative", title=valor),
            ],
        )
        .properties(title=titulo, height=320)
    )
    st.altair_chart(grafico, use_container_width=True)


def clasificar_z(z_score):
    if z_score >= 2:
        return "Muy por encima del promedio"
    if z_score >= 1:
        return "Por encima del promedio"
    if z_score <= -2:
        return "Muy por debajo del promedio"
    if z_score <= -1:
        return "Por debajo del promedio"
    return "Dentro del rango normal"


def aplicar_metodologia_z(datos, columna):
    datos_z = datos.copy()
    media = datos_z[columna].mean()
    desviacion = datos_z[columna].std(ddof=0)

    if desviacion == 0:
        datos_z["z_score"] = 0.0
    else:
        datos_z["z_score"] = (datos_z[columna] - media) / desviacion

    datos_z["clasificacion_z"] = datos_z["z_score"].apply(clasificar_z)
    return datos_z, media, desviacion


def mostrar_grafico_z(datos):
    datos_grafico = datos[["encuesta", "z_score"]].copy()
    base = alt.Chart(datos_grafico).encode(x=alt.X("encuesta:Q", title="Encuesta"))

    linea = base.mark_line(point=True).encode(
        y=alt.Y("z_score:Q", title="Indice de desviacion"),
        tooltip=[
            alt.Tooltip("encuesta:Q", title="Encuesta"),
            alt.Tooltip("z_score:Q", title="Indice", format=".2f"),
        ],
    )

    limites = pd.DataFrame(
        {
            "z_score": [-2, -1, 0, 1, 2],
            "limite": ["-2", "-1", "0", "1", "2"],
        }
    )
    reglas = (
        alt.Chart(limites)
        .mark_rule(strokeDash=[6, 4])
        .encode(
            y="z_score:Q",
            color=alt.Color("limite:N", legend=alt.Legend(title="Referencia")),
        )
    )

    st.altair_chart((linea + reglas).properties(height=320), use_container_width=True)


def leer_respuestas_docente(teacher_id):
    partes = []
    teacher_id = str(teacher_id)

    for chunk in pd.read_csv(
        DATA_PATH,
        names=COLUMNAS_CSV,
        dtype=str,
        chunksize=200_000,
    ):
        filtrado = chunk[chunk["teacher_id"] == teacher_id]
        if not filtrado.empty:
            partes.append(filtrado.copy())

    if not partes:
        return pd.DataFrame(columns=COLUMNAS_CSV)

    return pd.concat(partes, ignore_index=True)


def construir_resultados_docente(datos_docente, modelo):
    datos_docente = datos_docente.copy()
    datos_docente["valor"] = datos_docente["value_item"].map(CONVERSION)
    datos_docente = datos_docente.dropna(subset=["valor"])
    datos_docente["valor"] = datos_docente["valor"].astype(int)

    resultados = []
    grupos = datos_docente.groupby(["teacher_id", "student_id", "course_id"])

    for clave, grupo in grupos:
        teacher_id, student_id, course_id = clave
        respuestas = grupo["valor"].tolist()
        promedio, variabilidad, uniformidad, resultado = analizar_respuestas(respuestas)
        promedio_predicho = predecir(modelo, variabilidad, uniformidad)
        resultados.append(
            {
                "teacher_id": teacher_id,
                "student_id": student_id,
                "course_id": course_id,
                "promedio": promedio,
                "variabilidad": variabilidad,
                "uniformidad": uniformidad,
                "promedio_predicho": promedio_predicho,
                "resultado": resultado,
            }
        )

    return pd.DataFrame(resultados)


st.set_page_config(
    page_title="Promedio esperado docente",
    layout="centered",
)

st.title("Promedio esperado docente")

if not MODEL_PATH.exists():
    st.error(f"No se encontro el modelo en: {MODEL_PATH}")
    st.stop()

modelo = cargar_modelo()

st.subheader("Consultar docente desde el CSV")

if not DATA_PATH.exists():
    st.warning("No se encontro respuestas_analisis2.csv junto a la app.")
else:
    teacher_id = st.text_input("ID del docente", value="4241")

    if st.button("Calcular promedio esperado", type="primary"):
        if not teacher_id.strip():
            st.error("Ingrese un ID de docente.")
        else:
            with st.spinner("Buscando respuestas del docente en el CSV..."):
                respuestas_docente = leer_respuestas_docente(teacher_id.strip())

            if respuestas_docente.empty:
                st.warning("No existen datos para ese docente.")
            else:
                resultados_docente = construir_resultados_docente(
                    respuestas_docente,
                    modelo,
                )

                if resultados_docente.empty:
                    st.warning("El docente existe, pero no tiene respuestas validas A/B/C/D/E.")
                else:
                    promedio_futuro = resultados_docente["promedio_predicho"].mean()
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Encuestas", f"{len(resultados_docente)}")
                    col2.metric("Promedio esperado", f"{promedio_futuro:.2f}/5")
                    col3.metric("Interpretacion", interpretar_promedio(promedio_futuro))

                    st.subheader("Graficos del docente")
                    col_grafico1, col_grafico2 = st.columns(2)

                    with col_grafico1:
                        conteo_resultados = (
                            resultados_docente["resultado"]
                            .value_counts()
                            .rename_axis("resultado")
                            .reset_index(name="cantidad")
                        )
                        mostrar_grafico_pastel(
                            conteo_resultados,
                            "resultado",
                            "cantidad",
                            "Distribucion de resultados",
                        )

                    with col_grafico2:
                        conteo_respuestas = (
                            respuestas_docente["value_item"]
                            .value_counts()
                            .sort_index()
                            .rename_axis("respuesta")
                            .reset_index(name="cantidad")
                        )
                        mostrar_grafico_pastel(
                            conteo_respuestas,
                            "respuesta",
                            "cantidad",
                            "Distribucion de respuestas",
                        )

                    linea_docente = resultados_docente[
                        ["promedio", "promedio_predicho"]
                    ].reset_index(drop=True)
                    linea_docente = linea_docente.rename(
                        columns={"promedio_predicho": "promedio_esperado"}
                    )
                    linea_docente.index = linea_docente.index + 1
                    linea_docente.index.name = "encuesta"
                    st.line_chart(linea_docente)

                    resultados_docente_z, media_z, desviacion_z = aplicar_metodologia_z(
                        resultados_docente,
                        "promedio_predicho",
                    )
                    resultados_docente_z = resultados_docente_z.reset_index(drop=True)
                    resultados_docente_z["encuesta"] = resultados_docente_z.index + 1

                    st.subheader("Analisis de desviacion")
                    col_z1, col_z2 = st.columns(2)
                    col_z1.metric("Media", f"{media_z:.2f}")
                    col_z2.metric("Desviacion estandar", f"{desviacion_z:.2f}")
                    mostrar_grafico_z(resultados_docente_z)

                    conteo_z = (
                        resultados_docente_z["clasificacion_z"]
                        .value_counts()
                        .rename_axis("clasificacion")
                        .reset_index(name="cantidad")
                    )
                    mostrar_grafico_pastel(
                        conteo_z,
                        "clasificacion",
                        "cantidad",
                        "Clasificacion por desviacion",
                    )

                    st.dataframe(
                        resultados_docente_z.rename(
                                columns={
                                    "promedio_predicho": "promedio_esperado",
                                    "z_score": "indice_desviacion",
                                    "clasificacion_z": "clasificacion",
                                }
                        )[
                                [
                                    "teacher_id",
                                    "student_id",
                                    "course_id",
                                    "promedio",
                                    "promedio_esperado",
                                    "indice_desviacion",
                                    "clasificacion",
                                    "resultado",
                            ]
                        ].head(100),
                        use_container_width=True,
                        hide_index=True,
                    )
