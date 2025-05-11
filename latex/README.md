# Sistema de Generación de Informes en LaTeX

Este directorio contiene una estructura para generar informes en LaTeX para la Tarea 2 de GOP.

## Estructura del Proyecto

````
latex/
├── main.tex               # Archivo principal
├── preamble.tex           # Preámbulo con configuración y paquetes
├── portada.tex            # Portada del informe
├── ParteA/                # Archivos de la Parte A
│   ├── parteA.tex
│   ├── pregunta1.tex
│   ├── pregunta2.tex
│   ├── pregunta3.tex
│   └── resources/         # Recursos (tablas, gráficos) organizados por pregunta
│       ├── pregunta1/
│       ├── pregunta2/
│       ├── pregunta3/     # Tablas CSV y gráficos para pregunta 3
│       └── ...
├── ParteB/                # Archivos de la Parte B (misma estructura)
├── ParteC/                # Archivos de la Parte C (misma estructura)
├── templates/             # Plantillas para crear nuevos archivos
│   ├── csv_table_template.tex    # Ejemplos de importación de tablas CSV
│   ├── parte_template.tex
│   ├── pregunta_template.tex
│   └── README.md          # Documentación de templates

## Importación de tablas CSV

Para importar tablas desde archivos CSV directamente, usamos el paquete `csvsimple`. Este enfoque permite:
- Mantener los datos separados del código LaTeX
- Facilitar actualizaciones de datos sin tocar el código
- Organizar mejor el proyecto

### Ejemplo básico
```latex
\begin{table}[H]
    \centering
    \caption{Título de la tabla}
    \label{tab:ejemplo}
    \csvreader[
        tabular=ccc,                                         % formato de columnas (c=centrado)
        table head=\toprule \textbf{Col1} & \textbf{Col2} & \textbf{Col3} \\\midrule,
        command=#1 & $#2$ & $#3$,                           % formato por fila
        late after line=\\,
        table foot=\bottomrule
    ]{resources/pregunta3/datos.csv}{}{}    % Los {} vacíos son para el mapping de columnas y el contenido
\end{table}
````

### Para tablas grandes (que se salen de la página)

```latex
\begin{table}[H]
    \centering
    \caption{Título de la tabla grande}
    \label{tab:ejemplo_grande}
    \resizebox{\textwidth}{!}{                              % Ajusta automáticamente
        \csvsimple[...]{resources/pregunta3/datos_grandes.csv}
    }
\end{table}
```

├── templates/ # Plantillas para crear nuevos archivos
└── resources/ # Recursos globales del documento

````

## Mejoras Implementadas

### 1. Importación Directa de CSV a LaTeX

Hemos implementado un sistema para importar tablas CSV directamente en LaTeX sin necesidad de conversión manual:

- Se añadieron los paquetes `csvsimple`, `pgfplotstable` y `siunitx`
- Se crearon comandos personalizados para facilitar la importación de tablas
- Se implementó un sistema de ajuste automático para tablas grandes

### 2. Organización de Recursos

- Los recursos están organizados por pregunta en sus respectivas carpetas
- Se proporciona un script (`setup_latex_dirs.py`) para crear automáticamente la estructura de directorios

### 3. Plantillas para Estandarizar el Formato

- Plantillas para cada tipo de archivo (preguntas, partes, tablas CSV)
- Ejemplos detallados en `templates/csv_table_template.tex`

## Comandos Personalizados para Tablas

```latex
% Importar tabla CSV básica:
\importCSV{resources/preguntaN/datos.csv}{ccccc}{
    \textbf{Col1} & \textbf{Col2} & \textbf{Col3} & \textbf{Col4} & \textbf{Col5}
}{
    #1 & \$\num{#2}\$ & \$\num{#3}\$ & \$\num{#4}\$ & \$\num{#5}\$
}

% Importar tabla grande con ajuste automático:
\adjustableTable{0.95\textwidth}{Título de tabla grande}{
    \importCSV{...}{...}{...}{...}
}

% Tablas extremadamente anchas:
\begin{landscape}
    \adjustableTable{0.95\textwidth}{Título de tabla muy ancha}{...}
\end{landscape}
````

## Uso del Script de Configuración

El script `setup_latex_dirs.py` facilita la preparación de la estructura de directorios:

```bash
# Para crear la estructura de directorios
./setup_latex_dirs.py --create-dirs

# Para copiar archivos CSV y gráficos desde desarrollo a latex
./setup_latex_dirs.py --copy-files

# Para ver todas las opciones
./setup_latex_dirs.py --help
```

## Recomendaciones para el Equipo

1. **No convertir manualmente tablas CSV**: Usar los comandos personalizados para importarlas directamente.

2. **Para tablas muy grandes**:

   - Usar `\adjustableTable` para ajuste automático de tamaño
   - Para tablas extremadamente anchas, usar el entorno `landscape`

3. **Organización**:

   - Mantener cada pregunta en su propio archivo .tex
   - Colocar todos los recursos de cada pregunta en su carpeta correspondiente

4. **Uso de plantillas**:
   - Revisar las plantillas disponibles en el directorio `templates/`
   - El archivo `csv_table_template.tex` contiene ejemplos detallados para importar tablas

## Compilación del Documento

Para compilar el documento completo:

```bash
pdflatex main.tex
pdflatex main.tex  # Segunda pasada para referencias
```

## Contacto

Si hay dudas sobre la estructura o la implementación, consultar con el encargado de la configuración LaTeX.
