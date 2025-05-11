# Templates para LaTeX

Este directorio contiene plantillas para trabajar con el informe en LaTeX.

## Archivos principales

- `tarea_template.tex`: Estructura básica para el documento principal
- `parte_template.tex`: Plantilla para cada parte del documento (A, B, C)
- `pregunta_template.tex`: Plantilla para cada pregunta individual
- `preamble_template.tex`: Plantilla para el preámbulo con paquetes y configuraciones
- `portada_template.tex`: Plantilla para la portada del informe
- `csv_table_template.tex`: Ejemplos de cómo importar tablas CSV directamente

## Importación de tablas CSV

Hemos agregado funcionalidad para importar tablas CSV directamente, sin necesidad de convertir manualmente a LaTeX:

1. Los comandos básicos para importar CSV son:
   - `\csvreader[opciones]{archivo.csv}{}{}`: Comando general para importar CSV
   - `\csvautotabular{archivo.csv}`: Versión simplificada para tablas sencillas
   - `\adjustableTable{ancho}{título}{contenido}`: Para tablas que necesitan ajuste automático de tamaño

2. Los paquetes necesarios para esto son:
   - `csvsimple`: Importación básica de CSV (provee los comandos \csvreader y \csvautotabular)
   - `pgfplotstable`: Más opciones de formato para tablas
   - `siunitx`: Formateo de números (para usar \num{} en valores numéricos)

3. Ver ejemplos en el archivo `csv_table_template.tex`
3. Ver el archivo `csv_table_template.tex` para ejemplos prácticos

## Estructura de carpetas

Para cada parte (A, B, C), recomendamos:

1. Crear subcarpetas dentro de `resources/` para cada pregunta
   - Por ejemplo: `ParteA/resources/pregunta1/`, `ParteA/resources/pregunta2/`, etc.
2. Para tablas muy grandes que podrían salirse de la página:
   - Usar `\adjustableTable` que ajusta automáticamente el tamaño
   - O usar el entorno `landscape` para tablas extremadamente anchas

## Ejemplo de uso

```latex
% Importar una tabla CSV básica
\begin{table}[H]
    \centering
    \caption{Título de la tabla}
    \label{tab:ejemplo}
    \importCSV{resources/pregunta1/datos.csv}{ccccc}{
        \textbf{Col1} & \textbf{Col2} & \textbf{Col3} & \textbf{Col4} & \textbf{Col5}
    }{
        #1 & \$\num{#2}\$ & \$\num{#3}\$ & \$\num{#4}\$ & \$\num{#5}\$
    }
\end{table}

% Para tablas grandes con ajuste automático
\adjustableTable{0.95\textwidth}{Título de tabla grande}{
    \importCSV{resources/pregunta2/datos_grandes.csv}{ccccccccc}{
        \textbf{Col1} & \textbf{Col2} & \textbf{Col3} & \textbf{Col4} & \textbf{Col5} & \textbf{Col6} & \textbf{Col7} & \textbf{Col8} & \textbf{Col9}
    }{
        #1 & #2 & #3 & #4 & #5 & #6 & #7 & #8 & #9
    }
}
```
