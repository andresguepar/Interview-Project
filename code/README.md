# Interview Project - EEG and Eye Tracking Data Fusion

## Descripción
Este proyecto fusiona datos de EEG (Electroencefalografía) y Eye Tracking (Seguimiento Ocular) para análisis de investigación. Los scripts procesan archivos CSV de múltiples respondientes y generan datasets fusionados alineados por timestamp.

## Estructura del Proyecto
```
Interview Project/
├── Data/
│   └── Interview/
│       └── Folder 1/
│           ├── EEG - Folder 1(a)/          # Archivos EEG por respondiente
│           └── Eye Tracking - Folder 1(b)/  # Archivos Eye Tracking por respondiente
├── fused_respondents/                       # CSV fusionados por respondiente
├── fusion_data.py                          # Script principal de fusión
├── workflow.py                             # Workflow de análisis
└── README.md                               # Este archivo
```

## Scripts Principales

### fusion_data.py
Script principal que:
- Limpia metadatos de archivos CSV originales
- Fusiona datos EEG y Eye Tracking por respondiente
- Alinea datos por timestamp usando merge_asof
- Genera un CSV por respondiente en formato universal

**Uso:**
```bash
python fusion_data.py
```

**Salida:**
- Archivos `fused_[respondent_id].csv` en carpeta `fused_respondents/`

### workflow.py
Workflow de análisis que procesa los datos fusionados para:
- Análisis estadístico
- Visualizaciones
- Reportes de resultados

## Formato de Datos

### Entrada
- **EEG**: Archivos CSV con columnas `Row`, `Timestamp`, y canales EEG
- **Eye Tracking**: Archivos CSV con columnas `Row`, `Timestamp`, y métricas de seguimiento ocular

### Salida
- **CSV fusionados**: Un archivo por respondiente con datos EEG y Eye Tracking alineados por timestamp
- **Formato**: UTF-8, delimitador coma, sin metadatos

## Alineación de Timestamps
El script usa `pd.merge_asof()` para alinear datos por timestamp más cercano:
- Tolerancia: 1000 unidades (normalmente milisegundos)
- Dirección: 'nearest' (coincidencia más cercana)
- Ordenamiento: Por timestamp antes de fusionar

## Requisitos
- Python 3.7+
- pandas
- numpy (opcional, para análisis avanzado)

## Instalación
```bash
pip install pandas
```

## Uso
1. Coloca los archivos de datos en las carpetas correspondientes
2. Ejecuta `python fusion_data.py`
3. Revisa los archivos fusionados en `fused_respondents/`
4. Usa `workflow.py` para análisis adicionales

## Notas
- Los archivos originales deben tener la estructura de carpetas especificada
- Los metadatos se eliminan automáticamente durante el procesamiento
- Cada respondiente genera un archivo CSV fusionado independiente 