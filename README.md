# Tarea 2 - Gestión de Operaciones

Este repositorio contiene la solución a la Tarea 2 del curso de Gestión de Operaciones, enfocada en la modelación y resolución de un problema de optimización lineal para la Fundación Circular.

## Estructura del proyecto

El proyecto está organizado de la siguiente manera:

- `data/`: Contiene los archivos de datos utilizados en los modelos.

  - `datos_parametros.csv`: Parámetros generales del modelo.
  - `datos_periodos.csv`: Datos específicos para cada periodo del horizonte de planificación.

- `desarrollo/`: Contiene el código fuente y las respuestas a las preguntas.

  - `respuestas/`: Documentos markdown con las respuestas detalladas a cada pregunta.
  - `src/`: Código fuente para la resolución de los modelos.

- `docs/`: Documentación adicional y enunciado de la tarea.

## Preguntas abordadas

1. **Modelo de Optimización Lineal**: Formulación del modelo matemático para la planificación de la Fundación Circular.
2. **Resolución del Modelo Base**: Implementación y resolución del modelo utilizando software de optimización.
3. **Análisis de Falla**: Evaluación del impacto de falla de maquinaria en la planificación.
4. **Análisis de Adquisición**: Estudio de la conveniencia de adquirir nueva maquinaria.
5. **Análisis de Dotación Variable**: Evaluación del impacto de modificar la dotación de trabajadores.
6. **Análisis de Sensibilidad**: Estudio de la sensibilidad del modelo a variaciones en los parámetros.

## Requisitos

Para ejecutar los modelos se requiere:

- Python 3.x
- PuLP (para la optimización lineal)
- pandas, numpy, matplotlib, seaborn (para análisis de datos y visualizaciones)
- tabulate (para formateo de tablas)

## Ejecución

Cada modelo puede ejecutarse de forma independiente desde la carpeta `desarrollo/src`. Para más detalles, consulte el README en dicha carpeta.
