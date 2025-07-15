# Code - EEG & Eye Tracking Data Fusion

Este directorio contiene los scripts principales para la fusión y análisis de datos de EEG y Eye Tracking utilizados en el proyecto Interview Project.

## Estructura

```
code/
├── fusion_data.py   # Script para fusionar archivos EEG y Eye Tracking por respondiente
├── workflow.py      # Script de análisis y procesamiento posterior de los datos fusionados
├── README.md        # Este archivo
└── requirements.txt # Dependencias necesarias para ejecutar los scripts
```

## Descripción de los scripts

### fusion_data.py
- Limpia los archivos originales de EEG y Eye Tracking (elimina metadatos y deja solo la cabecera real y los datos).
- Fusiona los datos de ambos orígenes por respondiente, alineando por timestamp más cercano.
- Genera un archivo CSV por respondiente en formato universal (UTF-8, delimitador coma) en la carpeta `fused_respondents`.
- No requiere base de datos ni pasos manuales adicionales.

**Uso:**
```bash
python fusion_data.py
```

### workflow.py
- Permite realizar análisis estadístico, visualizaciones y reportes sobre los datos fusionados generados por `fusion_data.py`.
- Puedes personalizarlo para tus necesidades de análisis.

**Uso:**
```bash
python workflow.py
```

## Requisitos
- Python 3.7+
- pandas
- numpy (opcional, para análisis avanzado)

Instala las dependencias con:
```bash
pip install -r requirements.txt
```

## Notas
- Los archivos de entrada deben estar en la estructura de carpetas esperada (ver README principal del proyecto).
- Los archivos fusionados se generan en `fused_respondents/`.
- El formato de salida es compatible con cualquier software que acepte CSV estándar.

---

¿Dudas o sugerencias? Edita este README o abre un issue en el repositorio principal. 