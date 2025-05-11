# Análisis del Problema: Fundación Circular

## Descripción del problema

La Fundación Circular ha lanzado una iniciativa nacional para reacondicionar ropa donada y entregarla a comunidades vulnerables. El desafío consiste en implementar una operación eficiente en una planta que operará durante un horizonte de planificación de T periodos consecutivos.

## Componentes clave del sistema

### Flujos de entrada

- **Ropa en buen estado** (kb<sub>t</sub>): Puede ser entregada directamente
- **Ropa en mal estado** (km<sub>t</sub>): Debe ser transformada en género textil

### Procesos operativos

1. **Transformación**: Convertir ropa en mal estado a género textil
2. **Confección**: Fabricar nuevas prendas a partir del género textil
3. **Distribución**: Entregar prendas para satisfacer la demanda

### Recursos

- **Trabajadores contratados**: Dotación inicial w<sub>0</sub>
- **Trabajadores por boleta**: Contratables cada periodo
- **Capacidad de almacenamiento**: Limitada a s kilogramos

## Parámetros del modelo

### Parámetros temporales

- **T**: Horizonte de planificación (periodos)
- **kb<sub>t</sub>**: Kilogramos de ropa en buen estado que llegan en periodo t
- **km<sub>t</sub>**: Kilogramos de ropa en mal estado que llegan en periodo t
- **d<sub>t</sub>**: Demanda de prendas para el periodo t

### Parámetros de inventario

- **rb**: Inventario inicial de ropa en buen estado (kg)
- **rm**: Inventario inicial de ropa en mal estado (kg)
- **p**: Peso promedio de cada unidad de ropa (kg)
- **s**: Capacidad máxima de almacenamiento (kg)

### Parámetros de costos

- **cp**: Costo de penalización por demanda no satisfecha ($/prenda)
- **ct**: Costo por trabajador contratado por boleta ($/persona/periodo)
- **g**: Costo unitario de transformación a género ($/kg)
- **n**: Costo unitario de producción de prendas desde género ($/kg)
- **a**: Costo de almacenamiento ($/kg/periodo)
- **cc**: Costo por hora normal trabajada ($/hora)

### Parámetros de mano de obra

- **w<sub>0</sub>**: Dotación inicial de trabajadores
- **h**: Horas de trabajo por trabajador por periodo
- **τ<sub>g</sub>**: Horas-hombre para transformar 1 kg de ropa en mal estado a género
- **τ<sub>n</sub>**: Horas-hombre para confeccionar 1 kg de ropa reutilizada desde género

## Preguntas a resolver

1. **Modelado básico**: Desarrollar modelo de optimización lineal completo
2. **Implementación y resolución**: Determinar planificación óptima y costos
3. **Análisis de escenario - Falla técnica**: Evaluar impacto del aumento del 25% en tiempo de transformación
4. **Análisis de escenario - Adquisición de género**: Evaluar conveniencia de comprar género adicional
5. **Análisis de escenario - Dotación mínima**: Evaluar impacto de mantener un mínimo de trabajadores
6. **Análisis de sensibilidad**: Evaluar impacto de variaciones en la demanda (80% y 120%)

## Consideraciones iniciales para el modelado

### Variables de decisión potenciales

- Cantidad de ropa en buen estado a entregar por periodo
- Cantidad de ropa en mal estado a transformar por periodo
- Cantidad de género a utilizar para fabricar prendas por periodo
- Inventario de cada tipo de material al final de cada periodo
- Número de trabajadores por boleta a contratar por periodo
- Demanda no satisfecha por periodo

### Restricciones principales

- Balance de inventario para cada tipo de material
- Límite de capacidad de almacenamiento
- Límite de horas-hombre disponibles por periodo
- Cumplimiento o penalización de demanda
- Relación entre peso procesado y unidades producidas

### Función objetivo

Minimizar el costo total, incluyendo:

- Costos de personal (contratado y por boleta)
- Costos de procesamiento (transformación y producción)
- Costos de almacenamiento
- Costos de penalización por demanda no satisfecha

## Próximos pasos

1. Formalizar modelo matemático completo
2. Implementar en el software de optimización seleccionado
3. Realizar experimentos con los distintos escenarios
4. Analizar y documentar los resultados
