# Pregunta 1: Modelo de Optimización Lineal

## Planteamiento de la pregunta

> Modele el problema mediante optimización lineal, explicando el significado de parámetros, variables, función objetivo y restricciones utilizadas.

## Desarrollo de la respuesta

### Contexto del problema

La Fundación Circular ha lanzado una iniciativa para reacondicionar ropa donada y entregarla a comunidades vulnerables. El problema se centra en la gestión eficiente de una planta que:

- Recibe donaciones de ropa en buen y mal estado
- Transforma ropa en mal estado en género textil
- Produce nuevas prendas a partir del género textil
- Satisface demandas de prendas en cada periodo

El objetivo es minimizar los costos totales de operación sobre un horizonte de planificación definido, considerando costos de personal, procesamiento, almacenamiento y penalizaciones por demandas no satisfechas.

### Conjuntos e índices

- $T$: Conjunto de periodos del horizonte de planificación, indexado por $t \in \{1,2,...,T\}$

### Parámetros

#### Parámetros de entrada de materiales

- $kb_t$: Kilogramos de ropa en buen estado que llegan en el periodo $t$
- $km_t$: Kilogramos de ropa en mal estado que llegan en el periodo $t$
- $rb$: Inventario inicial de ropa en buen estado (kg)
- $rm$: Inventario inicial de ropa en mal estado (kg)
- $p$: Peso promedio de cada unidad de ropa (kg/prenda)

#### Parámetros de demanda

- $d_t$: Demanda de prendas para el periodo $t$
- $cp$: Costo de penalización por demanda no satisfecha ($/prenda)

#### Parámetros de costos

- $ct$: Costo por trabajador contratado por boleta ($/persona/periodo)
- $g$: Costo unitario de transformación a género ($/kg)
- $n$: Costo unitario de producción de prendas desde género ($/kg)
- $a$: Costo de almacenamiento ($/kg/periodo)
- $cc$: Costo por hora normal trabajada ($/hora)

#### Parámetros de capacidad

- $s$: Capacidad máxima de almacenamiento (kg)
- $w_0$: Dotación inicial de trabajadores
- $h$: Horas de trabajo por trabajador por periodo
- $\tau_g$: Horas-hombre para transformar 1 kg de ropa en mal estado a género
- $\tau_n$: Horas-hombre para confeccionar 1 kg de ropa reutilizada desde género

### Variables de decisión

#### Variables de procesamiento y producción

- $X_{t}$: Kilogramos de ropa en buen estado utilizados para satisfacer demanda en periodo $t$

  - Justificación: Representa el flujo principal de ropa en buen estado que se destina directamente para satisfacer la demanda.
  - Unidades: Kilogramos (kg)
  - Dominio: $X_t \geq 0$ (variable continua no negativa)

- $Y_{t}$: Kilogramos de ropa en mal estado transformados a género en periodo $t$

  - Justificación: Representa la cantidad de ropa en mal estado que se procesa para obtener género textil.
  - Unidades: Kilogramos (kg)
  - Dominio: $Y_t \geq 0$ (variable continua no negativa)

- $Z_{t}$: Kilogramos de género utilizados para fabricar prendas en periodo $t$
  - Justificación: Representa la cantidad de género que se utiliza para producir prendas reutilizadas.
  - Unidades: Kilogramos (kg)
  - Dominio: $Z_t \geq 0$ (variable continua no negativa)

#### Variables de inventario

- $IB_{t}$: Inventario de ropa en buen estado al final del periodo $t$

  - Justificación: Control del stock de ropa en buen estado disponible entre periodos.
  - Unidades: Kilogramos (kg)
  - Dominio: $IB_t \geq 0$ (variable continua no negativa)

- $IM_{t}$: Inventario de ropa en mal estado al final del periodo $t$

  - Justificación: Control del stock de ropa en mal estado disponible entre periodos.
  - Unidades: Kilogramos (kg)
  - Dominio: $IM_t \geq 0$ (variable continua no negativa)

- $IG_{t}$: Inventario de género al final del periodo $t$
  - Justificación: Control del stock de género textil disponible entre periodos.
  - Unidades: Kilogramos (kg)
  - Dominio: $IG_t \geq 0$ (variable continua no negativa)

#### Variables de recursos humanos

- $W_{t}$: Número de trabajadores por boleta contratados en periodo $t$
  - Justificación: Representa la fuerza laboral adicional contratada para satisfacer la demanda de mano de obra.
  - Unidades: Personas (trabajadores)
  - Dominio: $W_t \geq 0$, entero (variable entera no negativa)

#### Variables de satisfacción de demanda

- $NS_{t}$: Demanda no satisfecha en periodo $t$
  - Justificación: Mide la cantidad de demanda que no puede ser cubierta durante el periodo.
  - Unidades: Prendas
  - Dominio: $NS_t \geq 0$ (variable continua no negativa)

### Función objetivo

Minimizar el costo total de la operación:

$$\min Z = \sum_{t=1}^{T} \left[ W_t \cdot ct + cc \cdot h \cdot w_0 + g \cdot Y_t + n \cdot Z_t + a \cdot (IB_t + IM_t + IG_t) + cp \cdot NS_t \right]$$

Esta función objetivo integra todos los componentes de costo que la Fundación Circular busca minimizar:

#### Desglose de la función objetivo

1. **Costos de personal**:

   - $W_t \cdot ct$: Costo de los trabajadores contratados por boleta en cada periodo $t$
   - $cc \cdot h \cdot w_0$: Costo fijo de la dotación inicial de trabajadores contratados
   - Justificación: Representa el gasto en recursos humanos, diferenciando entre personal fijo y temporal

2. **Costos de procesamiento**:

   - $g \cdot Y_t$: Costo de transformar ropa en mal estado a género textil
   - $n \cdot Z_t$: Costo de producir prendas a partir de género textil
   - Justificación: Captura los costos variables asociados a los procesos productivos principales

3. **Costos de almacenamiento**:

   - $a \cdot (IB_t + IM_t + IG_t)$: Costo de mantener inventarios de los tres tipos de materiales
   - Justificación: Refleja los costos de mantener stock entre periodos, aplicando el mismo costo unitario a todos los tipos de materiales conforme al enunciado

4. **Costos de penalización**:
   - $cp \cdot NS_t$: Penalización por demanda no satisfecha
   - Justificación: Incorpora el costo social/económico de no entregar las prendas comprometidas

### Restricciones

#### Restricciones de balance de inventario

Estas restricciones garantizan la conservación del flujo de materiales entre periodos consecutivos.

##### Balance de inventario de ropa en buen estado

$$IB_t = IB_{t-1} + kb_t - X_t \quad \forall t \in T$$

Donde:

- $IB_0 = rb$ (inventario inicial de ropa en buen estado)

**Interpretación**: El inventario de ropa en buen estado al final del periodo $t$ es igual al inventario del periodo anterior, más las llegadas de ropa en buen estado en el periodo actual, menos la cantidad utilizada para satisfacer demanda.

##### Balance de inventario de ropa en mal estado

$$IM_t = IM_{t-1} + km_t - Y_t \quad \forall t \in T$$

Donde:

- $IM_0 = rm$ (inventario inicial de ropa en mal estado)

**Interpretación**: El inventario de ropa en mal estado al final del periodo $t$ es igual al inventario del periodo anterior, más las llegadas de ropa en mal estado en el periodo actual, menos la cantidad transformada a género.

##### Balance de inventario de género

$$IG_t = IG_{t-1} + Y_t - Z_t \quad \forall t \in T$$

Donde:

- $IG_0 = 0$ (se asume que no hay inventario inicial de género)

**Interpretación**: El inventario de género textil al final del periodo $t$ es igual al inventario del periodo anterior, más la cantidad producida por transformación de ropa en mal estado, menos la cantidad utilizada para fabricar prendas.

#### Restricción de capacidad de almacenamiento

$$IB_t + IM_t + IG_t \leq s \quad \forall t \in T$$

**Interpretación**: La suma de todos los inventarios al final de cada periodo no puede exceder la capacidad máxima de almacenamiento disponible ($s$ kg). Esta restricción asegura que la fundación no sobrepase su infraestructura de almacenamiento.

#### Restricción de disponibilidad de horas-hombre

$$\tau_g \cdot Y_t + \tau_n \cdot Z_t \leq h \cdot (w_0 + W_t) \quad \forall t \in T$$

**Interpretación**: El tiempo total requerido para las operaciones de transformación de ropa a género ($\tau_g \cdot Y_t$) y fabricación de prendas desde género ($\tau_n \cdot Z_t$) no puede exceder la disponibilidad total de horas-hombre, determinada por el número total de trabajadores (contratados $w_0$ y por boleta $W_t$) multiplicado por las horas disponibles por trabajador ($h$).

#### Restricción de satisfacción de la demanda

$$\frac{X_t}{p} + \frac{Z_t}{p} + NS_t \geq d_t \quad \forall t \in T$$

**Interpretación**: La demanda de prendas en cada periodo debe ser satisfecha mediante:

- Prendas de ropa en buen estado ($\frac{X_t}{p}$ prendas, considerando que cada prenda pesa $p$ kg)
- Prendas fabricadas a partir de género ($\frac{Z_t}{p}$ prendas)
- Demanda no satisfecha ($NS_t$ prendas), que conlleva una penalización

#### Restricciones de dominio de variables

$$X_t, Y_t, Z_t, IB_t, IM_t, IG_t \geq 0 \quad \forall t \in T$$

**Interpretación**: Las variables de flujo e inventario son continuas y no negativas.

$$W_t \geq 0, \text{ entero} \quad \forall t \in T$$

**Interpretación**: El número de trabajadores por boleta debe ser un valor entero no negativo, ya que no se pueden contratar fracciones de personas.

$$NS_t \geq 0 \quad \forall t \in T$$

**Interpretación**: La demanda no satisfecha debe ser no negativa.
