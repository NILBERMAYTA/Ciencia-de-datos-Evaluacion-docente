# Proyecto Ciencia de Datos

Aplicacion en Streamlit para consultar el promedio esperado de docentes usando un modelo de regresion guardado en `modelo_regresion_docentes.pkl`.

## Requisitos

- Python 3.13
- Archivo `modelo_regresion_docentes.pkl`
- Archivo `respuestas_analisis2.csv`

## Instalacion

Crear el entorno virtual:

```powershell
python -m venv .venv
```

Activar el entorno virtual:

```powershell
.\.venv\Scripts\Activate.ps1
```

Instalar dependencias:

```powershell
pip install -r requirements.txt
```

## Ejecucion

Ejecutar la aplicacion:

```powershell
streamlit run app.py
```

Abrir en el navegador:

```text
http://localhost:8501
```

## Uso

1. Ingresar el ID del docente.
2. Presionar `Calcular promedio esperado`.
3. Revisar el promedio esperado, la interpretacion, los graficos y la tabla de resultados.

## Archivos principales

- `app.py`: interfaz grafica en Streamlit.
- `requirements.txt`: librerias necesarias.
- `modelo_regresion_docentes.pkl`: modelo entrenado.
- `respuestas_analisis2.csv`: datos de respuestas usados para consultar docentes.
