# Resolución del Modelo Base (Pregunta 2)

Esta carpeta contiene la implementación del modelo de optimización lineal base para la Fundación Circular, tal como se plantea en la Pregunta 2.

## Archivos

- `resolver_modelo_pregunta2.py`: Script principal que implementa y resuelve el modelo de programación lineal utilizando PuLP. Este script ahora incluye también las funciones para generar visualizaciones mejoradas.
- `grafico_*_p2.png`: Visualizaciones generadas por el modelo que incluyen características mejoradas.
- `resultados_tabla*.csv`: Tablas con los resultados detallados del modelo.

## Visualizaciones Mejoradas

El script principal ahora genera visualizaciones de alta calidad que ofrecen:

- Mayor claridad en la presentación de datos
- Información adicional como porcentajes y valores absolutos
- Análisis comparativo entre diferentes métricas
- Resúmenes estadísticos integrados directamente en los gráficos
- Paletas de colores optimizadas para mejor interpretación visual

## Resultados

Los resultados completos se presentan en el archivo `/desarrollo/respuestas/pregunta2_resolucion.md`.

Las visualizaciones mejoradas han sido incorporadas directamente al documento de resultados.

## Ejecución

Para ejecutar el modelo y generar todas las visualizaciones mejoradas, desde el directorio raíz del proyecto:

```bash
python desarrollo/src/pregunta2/resolver_modelo_pregunta2.py
```

> **Nota:** El archivo `mejorar_graficos.py` ya no es necesario, pues toda su funcionalidad ha sido integrada directamente en el script principal.
