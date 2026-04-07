# Bizum Analytics - Streamlit App

Aplicación web en Python con diseño estilo iOS 26 para analizar y visualizar registros de Bizums entre amigos.

## Requisitos

```bash
pip install streamlit pandas plotly scikit-learn
```

## Estructura

```
bizum_app/
├── app.py              # Aplicación Streamlit
├── data/               # ← Carpeta para los archivos CSV
│   └── bizums_data.csv
├── SPEC.md
└── README.md
```

## Uso

1. Coloca tus archivos `.csv` dentro de la carpeta `data/`
2. Ejecuta:

```bash
streamlit run app.py
```

3. Se abrirá en tu navegador en `http://localhost:8501`

## Formato CSV esperado

```csv
Fecha,Emisor,Receptor,Cantidad,Acumulado
2025-01-15,Tom,Edu,15.50,15.50
...
```

## Funcionalidades

- **Dashboard**: Resumen con métricas clave
- **Rankings**: Top enviadores/recibidores
- **Tendencias**: Mapas de calor y detección de picos
- **Proyecciones**: Forecasting con regresión lineal
- **Probabilidades**: Análisis condicional y matriz de afinidad

## Tema

El diseño usa CSS personalizado con estilo iOS 26:
- Glassmorphism
- Esquinas redondeadas
- Tipografía Inter
- Gradientes suaves
