# Código Fuente de la Tarea 2

Este directorio contiene todo el código fuente utilizado para resolver los problemas planteados en la Tarea 2 de la asignatura de Gestión de Operaciones.

## Estructura de directorios

El código está organizado según las preguntas de la tarea:

- `resolver_modelo_base.py`: Implementación base del modelo de optimización.
- `pregunta2/`: Resolución del modelo base (Pregunta 2).
- `pregunta3/`: Análisis de falla de maquinaria (Pregunta 3).
- `pregunta4/`: Análisis de adquisición de maquinaria (Pregunta 4).
- `pregunta5/`: Análisis de dotación variable (Pregunta 5).
- `pregunta6/`: Análisis de sensibilidad (Pregunta 6).

## Ejecución de los modelos

Cada carpeta contiene un script principal que puede ejecutarse de forma independiente para resolver la pregunta correspondiente. Para ejecutar cualquiera de los modelos:

```bash
# Para el modelo base
python resolver_modelo_base.py

# Para los modelos específicos de cada pregunta
python pregunta2/resolver_modelo_pregunta2.py
python pregunta3/resolver_modelo_pregunta3.py
# etc.
```

## Dependencias

Los scripts requieren las siguientes bibliotecas de Python:

- PuLP (para la optimización lineal)
- pandas (para el manejo de datos)
- numpy (para cálculos numéricos)
- matplotlib y seaborn (para visualizaciones)
- tabulate (para formateo de tablas)

Se pueden instalar con:

```bash
pip install pulp pandas numpy matplotlib seaborn tabulate
```
