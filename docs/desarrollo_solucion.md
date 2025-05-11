# Desarrollo de la Solución: Fundación Circular

## Estado actual del proyecto

### Fase 1: Modelo matemático (Completado)

- [x] Identificación de parámetros del problema
- [x] Definición formal de variables de decisión
- [x] Formulación de la función objetivo
- [x] Desarrollo de restricciones matemáticas
- [x] Validación del modelo matemático

### Fase 2: Implementación computacional (Pendiente)

- [ ] Selección del entorno de programación
- [ ] Implementación del modelo base
- [ ] Validación con datos de prueba
- [ ] Resolución del modelo base
- [ ] Documentación de resultados iniciales

### Fase 3: Análisis de escenarios alternativos (Pendiente)

- [ ] Escenario de falla técnica (aumento 25% en tiempo)
- [ ] Escenario de adquisición de género adicional
- [ ] Escenario de política de dotación mínima
- [ ] Análisis de sensibilidad para demanda (80% y 120%)

## Desarrollo del modelo matemático

El modelo matemático ha sido completado en el archivo [pregunta1_modelo.md](./pregunta1_modelo.md). A continuación se presenta un resumen de los componentes principales:

### Variables de decisión

Se han definido tres tipos principales de variables:

1. **Variables de procesamiento y producción**:

   - $X_t$: Kilogramos de ropa en buen estado utilizados para satisfacer demanda
   - $Y_t$: Kilogramos de ropa en mal estado transformados a género
   - $Z_t$: Kilogramos de género utilizados para fabricar prendas

2. **Variables de inventario**:

   - $IB_t$: Inventario de ropa en buen estado al final del periodo
   - $IM_t$: Inventario de ropa en mal estado al final del periodo
   - $IG_t$: Inventario de género al final del periodo

3. **Variables de recursos y demanda**:
   - $W_t$: Número de trabajadores por boleta contratados
   - $NS_t$: Demanda no satisfecha en prendas

### Función objetivo

La función objetivo minimiza los costos totales de operación:

$$\min Z = \sum_{t=1}^{T} \left[ W_t \cdot ct + cc \cdot h \cdot w_0 + g \cdot Y_t + n \cdot Z_t + a \cdot (IB_t + IM_t + IG_t) + cp \cdot NS_t \right]$$

Esta incluye costos de personal, procesamiento, almacenamiento y penalizaciones por demanda insatisfecha.

### Restricciones

Las principales restricciones del modelo son:

1. **Balances de inventario** para cada tipo de material:

   - Ropa en buen estado: $IB_t = IB_{t-1} + kb_t - X_t$
   - Ropa en mal estado: $IM_t = IM_{t-1} + km_t - Y_t$
   - Género textil: $IG_t = IG_{t-1} + Y_t - Z_t$

2. **Capacidad de almacenamiento**: $IB_t + IM_t + IG_t \leq s$

3. **Disponibilidad de horas-hombre**: $\tau_g \cdot Y_t + \tau_n \cdot Z_t \leq h \cdot (w_0 + W_t)$

4. **Satisfacción de demanda**: $\frac{X_t}{p} + \frac{Z_t}{p} + NS_t \geq d_t$

5. **Restricciones de no negatividad y dominio** para todas las variables

## Implementación computacional

_Pendiente de comenzar_

## Resultados preliminares

_Pendiente de obtener_

## Plan de trabajo actualizado

| Actividad                             | Estado     | Fecha estimada | Archivo asociado          |
| ------------------------------------- | ---------- | -------------- | ------------------------- |
| Completar modelo matemático           | Completado | 10/05/2025     | pregunta1_modelo.md       |
| Implementar modelo en software        | Pendiente  | -              | pregunta2_resolucion.md   |
| Resolver caso base                    | Pendiente  | -              | pregunta2_resolucion.md   |
| Analizar escenario de falla técnica   | Pendiente  | -              | pregunta3_falla.md        |
| Analizar escenario de adquisición     | Pendiente  | -              | pregunta4_adquisicion.md  |
| Analizar escenario de dotación mínima | Pendiente  | -              | pregunta5_dotacion.md     |
| Realizar análisis de sensibilidad     | Pendiente  | -              | pregunta6_sensibilidad.md |
| Documentar resultados finales         | Pendiente  | -              | -                         |

## Notas de avance

_Esta sección se irá actualizando con el progreso diario del proyecto_

- **10/05/2025**: Inicio del proyecto. Se ha realizado el análisis inicial del problema y se ha estructurado el plan de trabajo.
- **10/05/2025**: Completado el modelo de optimización lineal (Pregunta 1). Se ha desarrollado un modelo que incluye variables de decisión para procesamiento, inventarios y recursos humanos, con restricciones que capturan todas las limitaciones operativas relevantes. El modelo ha sido documentado detalladamente en el archivo `pregunta1_modelo.md` con justificación de cada componente y análisis de supuestos.
