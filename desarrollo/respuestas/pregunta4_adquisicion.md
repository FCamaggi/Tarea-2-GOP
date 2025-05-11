# Pregunta 4: Análisis de Escenario - Adquisición de Género Adicional

## Planteamiento de la pregunta

> Suponga que, debido a una alianza con otra organización, la fundación tiene la posibilidad de adquirir d kg adicionales de género ya transformado al inicio del primer periodo, pagando un valor de $cf por kilogramo. ¿Cuál sería el impacto de aceptar esta oferta en el resultado final? Agregue esta nueva condición al modelo, resuelva nuevamente y determine si resulta conveniente aceptar la propuesta considerando el costo total del sistema.

## Desarrollo de la respuesta

### Modificación del modelo

Para incorporar la posibilidad de adquirir género adicional, se introduce una nueva variable de decisión:

- **GA**: Kilogramos de género adicional adquiridos al inicio del primer periodo (≥ 0)

Esta variable está asociada a un nuevo parámetro:

- **cf**: Costo por kilogramo de género adicional adquirido ($)
- **d**: Cantidad máxima de género adicional disponible para adquirir (kg)

#### Modificaciones a la función objetivo

La función objetivo debe incluir el costo adicional de adquirir este género:

**Función objetivo original**:
$$\min Z = \sum_{t=1}^{T} \left[ W_t \cdot ct + cc \cdot h \cdot w_0 + g \cdot Y_t + n \cdot Z_t + a \cdot (IB_t + IM_t + IG_t) + cp \cdot NS_t \right]$$

**Función objetivo actualizada**:
$$\min Z = cf \cdot GA + \sum_{t=1}^{T} \left[ W_t \cdot ct + cc \cdot h \cdot w_0 + g \cdot Y_t + n \cdot Z_t + a \cdot (IB_t + IM_t + IG_t) + cp \cdot NS_t \right]$$

#### Modificaciones a las restricciones

La restricción de balance de inventario de género para el primer periodo debe actualizarse:

**Restricción original para t=1**:
$$IG_1 = 0 + Y_1 - Z_1$$

**Restricción actualizada para t=1**:
$$IG_1 = 0 + GA + Y_1 - Z_1$$

Adicionalmente, debe añadirse una restricción para limitar la cantidad de género adicional que se puede adquirir:

$$GA \leq d$$

La modificación en el código de implementación sería:

```python
# Nueva variable de decisión
GA = pulp.LpVariable("GA", lowBound=0)

# Actualización de la función objetivo
problem += cf * GA + pulp.lpSum([W[t]*ct + cc*h*w0 + g*Y[t] + n*Z[t] +
                                a*(IB[t] + IM[t] + IG[t]) + cp*NS[t] for t in range(T)])

# Actualización del balance de inventario de género para el primer periodo (t=0)
problem += IG[0] == GA + Y[0] - Z[0]

# Restricción sobre la cantidad máxima de género adicional
problem += GA <= d
```

### Análisis de conveniencia

Para determinar si es conveniente aceptar la oferta, debemos comparar:

1. El **costo marginal** de adquirir el género adicional (cf por kg)
2. El **beneficio marginal** que genera dicha adquisición

El beneficio marginal podría manifestarse como:

- Reducción en costos de transformación al no necesitar procesar tanta ropa en mal estado
- Reducción en costos de personal al requerir menos horas-hombre
- Reducción en costos de penalización al poder satisfacer más demanda
- Potencial reducción en costos de inventario al optimizar los flujos

La decisión será conveniente si y solo si el beneficio marginal supera al costo marginal.

### Resultados comparativos

#### Tabla 1: Valor óptimo de género adicional adquirido

| Parámetro                             | Valor             |
| ------------------------------------- | ----------------- |
| Género adicional adquirido (GA)       | [Valor óptimo] kg |
| Costo de adquisición (cf × GA)        | $ [Valor]         |
| % del máximo disponible (GA/d × 100%) | [Porcentaje] %    |

#### Tabla 2: Comparación de resultados generales

| Indicador                            | Modelo base         | Modelo con adquisición | Diferencia       | % Variación                    |
| ------------------------------------ | ------------------- | ---------------------- | ---------------- | ------------------------------ |
| Costo total                          | $ Z_base            | $ Z_adq                | $ Z_adq - Z_base | (Z_adq - Z_base)/Z_base × 100% |
| Ropa en mal estado procesada (total) | ΣY_t                | ΣY_t'                  | ΣY_t' - ΣY_t     | (ΣY_t' - ΣY_t)/ΣY_t × 100%     |
| Género utilizado (total)             | ΣZ_t                | ΣZ_t'                  | ΣZ_t' - ΣZ_t     | (ΣZ_t' - ΣZ_t)/ΣZ_t × 100%     |
| Trabajadores por boleta (total)      | ΣW_t                | ΣW_t'                  | ΣW_t' - ΣW_t     | (ΣW_t' - ΣW_t)/ΣW_t × 100%     |
| Demanda no satisfecha (total)        | ΣNS_t               | ΣNS_t'                 | ΣNS_t' - ΣNS_t   | (ΣNS_t' - ΣNS_t)/ΣNS_t × 100%  |
| Inventario promedio (total)          | Σ(IB_t+IM_t+IG_t)/T | Σ(IB_t'+IM_t'+IG_t')/T | Diferencia       | % Variación                    |

#### Tabla 3: Comparación detallada de costos

| Componente de costo                   | Modelo base ($)       | Modelo con adquisición ($) | Diferencia ($)                                 | % Ahorro/Incremento                                                 |
| ------------------------------------- | --------------------- | -------------------------- | ---------------------------------------------- | ------------------------------------------------------------------- |
| Adquisición de género                 | 0                     | cf × GA                    | cf × GA                                        | -                                                                   |
| Transformación a género               | g × ΣY_t              | g × ΣY_t'                  | g × (ΣY_t' - ΣY_t)                             | (ΣY_t' - ΣY_t)/ΣY_t × 100%                                          |
| Producción de prendas                 | n × ΣZ_t              | n × ΣZ_t'                  | n × (ΣZ_t' - ΣZ_t)                             | (ΣZ_t' - ΣZ_t)/ΣZ_t × 100%                                          |
| Personal contratado                   | cc × h × w₀ × T       | cc × h × w₀ × T            | 0                                              | 0%                                                                  |
| Personal por boleta                   | ct × ΣW_t             | ct × ΣW_t'                 | ct × (ΣW_t' - ΣW_t)                            | (ΣW_t' - ΣW_t)/ΣW_t × 100%                                          |
| Almacenamiento                        | a × Σ(IB_t+IM_t+IG_t) | a × Σ(IB_t'+IM_t'+IG_t')   | a × [Σ(IB_t'+IM_t'+IG_t') - Σ(IB_t+IM_t+IG_t)] | [Σ(IB_t'+IM_t'+IG_t') - Σ(IB_t+IM_t+IG_t)]/Σ(IB_t+IM_t+IG_t) × 100% |
| Penalización por demanda insatisfecha | cp × ΣNS_t            | cp × ΣNS_t'                | cp × (ΣNS_t' - ΣNS_t)                          | (ΣNS_t' - ΣNS_t)/ΣNS_t × 100%                                       |
| **Costo total**                       | **Z_base**            | **Z_adq**                  | **Z_adq - Z_base**                             | **(Z_adq - Z_base)/Z_base × 100%**                                  |

### Análisis costo-beneficio

#### 1. Análisis del costo directo

- **Costo de adquisición**: cf × GA
- **Ahorro en transformación**: g × (ΣY_t - ΣY_t')
- **Ahorro en mano de obra**: ct × (ΣW_t - ΣW_t')
- **Ahorro neto directo**: Ahorro en transformación + Ahorro en mano de obra - Costo de adquisición

#### 2. Análisis de beneficios operativos

- **Mejora en satisfacción de demanda**: Reducción en penalizaciones y aumento en servicio al cliente
- **Flexibilidad operativa**: Disponibilidad inmediata de material procesado sin necesidad de transformación
- **Posible reducción de inventarios**: Optimización de flujos de material a lo largo del horizonte

#### 3. Análisis de beneficios estratégicos

- **Establecimiento de relación con otra organización**: Valor de la alianza a largo plazo
- **Diversificación de fuentes de suministro**: Reducción de riesgos operativos
- **Posibilidad de acuerdos futuros**: Precedente para adquisiciones adicionales en otros periodos

## Conclusiones preliminares

Basado en los resultados del modelo y el análisis comparativo, podemos concluir:

1. **Conveniencia económica**: [Será conveniente / No será conveniente] aceptar la oferta de adquisición de género adicional desde una perspectiva puramente económica, ya que [el costo total del sistema disminuye / aumenta] en un [porcentaje]%.

2. **Impactos operativos**: La adquisición de género adicional permite [reducir la transformación de ropa en mal estado / optimizar la utilización de mano de obra / mejorar la satisfacción de demanda / otros impactos relevantes].

3. **Recomendación final**: La Fundación Circular debería [aceptar / rechazar] la propuesta de adquisición de género adicional, considerando tanto los aspectos económicos como los operativos y estratégicos.

La cuantificación exacta de estos impactos se determinará una vez resuelto el modelo actualizado con los parámetros específicos del problema.
