# Pregunta 5: Análisis de Escenario - Política de Dotación Mínima

## Planteamiento de la pregunta

> Considere ahora que la fundación ha establecido una nueva política operativa, la cual exige mantener al menos tr trabajadores activos en cada periodo del horizonte de planificación. Esta dotación mínima debe contemplar tanto a los trabajadores contratados como a aquellos que prestan servicios mediante boleta. Incorpore esta restricción al modelo y resuelva nuevamente. Compare los resultados obtenidos con los de la solución inicial y analice las consecuencias de esta política en los costos totales, inventarios y cumplimiento de la demanda.

## Desarrollo de la respuesta

### Modificación del modelo

La nueva política operativa establece un mínimo de tr trabajadores activos por periodo. Esto introduce un nuevo parámetro:

- **tr**: Número mínimo de trabajadores activos requeridos por periodo

Esta política se traduce en una nueva restricción que debe incorporarse al modelo:

**Nueva restricción**:
$$w_0 + W_t \geq tr \quad \forall t \in T$$

Donde:

- $w_0$ es la dotación inicial de trabajadores contratados
- $W_t$ es el número de trabajadores por boleta contratados en el periodo $t$

La modificación en el código de implementación sería:

```python
# Nuevo parámetro
tr = 10  # Ejemplo: mínimo de 10 trabajadores por periodo

# Nueva restricción de dotación mínima
for t in range(T):
    problem += w0 + W[t] >= tr
```

### Resultados comparativos

#### Tabla 1: Comparación de recursos humanos

| Periodo         | Trabajadores activos |                                | Horas disponibles |                                      | % Utilización horas-hombre        |                                      |
| --------------- | -------------------- | ------------------------------ | ----------------- | ------------------------------------ | --------------------------------- | ------------------------------------ |
|                 | **Base**             | **Dotación mín.**              | **Base**          | **Dotación mín.**                    | **Base**                          | **Dotación mín.**                    |
| 1               | w₀+W₁                | w₀+W₁'                         | h\*(w₀+W₁)        | h\*(w₀+W₁')                          | (τg*Y₁+τn*Z₁)/(h*(w₀+W₁))*100%    | (τg*Y₁'+τn*Z₁')/(h*(w₀+W₁'))*100%    |
| 2               | w₀+W₂                | w₀+W₂'                         | h\*(w₀+W₂)        | h\*(w₀+W₂')                          | (τg*Y₂+τn*Z₂)/(h*(w₀+W₂))*100%    | (τg*Y₂'+τn*Z₂')/(h*(w₀+W₂'))*100%    |
| ...             | ...                  | ...                            | ...               | ...                                  | ...                               | ...                                  |
| T               | w₀+W_T               | w₀+W_T'                        | h\*(w₀+W_T)       | h\*(w₀+W_T')                         | (τg*Y_T+τn*Z_T)/(h*(w₀+W_T))*100% | (τg*Y_T'+τn*Z_T')/(h*(w₀+W_T'))*100% |
| **Total**       | **w₀\*T+ΣW_t**       | **w₀\*T+ΣW_t'**                | **h*(w₀*T+ΣW_t)** | **h*(w₀*T+ΣW_t')**                   | **Promedio**                      | **Promedio**                         |
| **Variación**   | **-**                | **ΣW_t' - ΣW_t**               | **-**             | **h\*(ΣW_t' - ΣW_t)**                | **-**                             | **-**                                |
| **% Variación** | **-**                | **(ΣW_t' - ΣW_t)/ΣW_t × 100%** | **-**             | **h*(ΣW_t' - ΣW_t)/(h*ΣW_t) × 100%** | **-**                             | **-**                                |

#### Tabla 2: Comparación de producción y procesamiento

| Periodo         | Ropa en mal estado procesada (kg) |                                | Género utilizado (kg) |                                | Prendas producidas |                       |
| --------------- | --------------------------------- | ------------------------------ | --------------------- | ------------------------------ | ------------------ | --------------------- |
|                 | **Base**                          | **Dotación mín.**              | **Base**              | **Dotación mín.**              | **Base**           | **Dotación mín.**     |
| 1               | Y₁                                | Y₁'                            | Z₁                    | Z₁'                            | (X₁+Z₁)/p          | (X₁'+Z₁')/p           |
| 2               | Y₂                                | Y₂'                            | Z₂                    | Z₂'                            | (X₂+Z₂)/p          | (X₂'+Z₂')/p           |
| ...             | ...                               | ...                            | ...                   | ...                            | ...                | ...                   |
| T               | Y_T                               | Y_T'                           | Z_T                   | Z_T'                           | (X_T+Z_T)/p        | (X_T'+Z_T')/p         |
| **Total**       | **ΣY_t**                          | **ΣY_t'**                      | **ΣZ_t**              | **ΣZ_t'**                      | **(ΣX_t+ΣZ_t)/p**  | **(ΣX_t'+ΣZ_t')/p**   |
| **Variación**   | **-**                             | **ΣY_t' - ΣY_t**               | **-**                 | **ΣZ_t' - ΣZ_t**               | **-**              | **Variación total**   |
| **% Variación** | **-**                             | **(ΣY_t' - ΣY_t)/ΣY_t × 100%** | **-**                 | **(ΣZ_t' - ΣZ_t)/ΣZ_t × 100%** | **-**              | **% Variación total** |

#### Tabla 3: Comparación de inventarios

| Periodo         | Inventario total (kg)   |                              | % Capacidad utilizada    |                                 |
| --------------- | ----------------------- | ---------------------------- | ------------------------ | ------------------------------- |
|                 | **Base**                | **Dotación mín.**            | **Base**                 | **Dotación mín.**               |
| 1               | IB₁+IM₁+IG₁             | IB₁'+IM₁'+IG₁'               | (IB₁+IM₁+IG₁)/s\*100%    | (IB₁'+IM₁'+IG₁')/s\*100%        |
| 2               | IB₂+IM₂+IG₂             | IB₂'+IM₂'+IG₂'               | (IB₂+IM₂+IG₂)/s\*100%    | (IB₂'+IM₂'+IG₂')/s\*100%        |
| ...             | ...                     | ...                          | ...                      | ...                             |
| T               | IB_T+IM_T+IG_T          | IB_T'+IM_T'+IG_T'            | (IB_T+IM_T+IG_T)/s\*100% | (IB_T'+IM_T'+IG_T')/s\*100%     |
| **Promedio**    | **Σ(IB_t+IM_t+IG_t)/T** | **Σ(IB_t'+IM_t'+IG_t')/T**   | **Promedio Base**        | **Promedio Dotación mín.**      |
| **Variación**   | **-**                   | **Diferencia en promedio**   | **-**                    | **Diferencia en %**             |
| **% Variación** | **-**                   | **% Diferencia en promedio** | **-**                    | **% Diferencia en utilización** |

#### Tabla 4: Comparación de satisfacción de demanda

| Periodo         | Demanda satisfecha |                                       | Demanda no satisfecha |                                  | % Satisfacción              |                                       |
| --------------- | ------------------ | ------------------------------------- | --------------------- | -------------------------------- | --------------------------- | ------------------------------------- |
|                 | **Base**           | **Dotación mín.**                     | **Base**              | **Dotación mín.**                | **Base**                    | **Dotación mín.**                     |
| 1               | d₁-NS₁             | d₁-NS₁'                               | NS₁                   | NS₁'                             | (d₁-NS₁)/d₁\*100%           | (d₁-NS₁')/d₁\*100%                    |
| 2               | d₂-NS₂             | d₂-NS₂'                               | NS₂                   | NS₂'                             | (d₂-NS₂)/d₂\*100%           | (d₂-NS₂')/d₂\*100%                    |
| ...             | ...                | ...                                   | ...                   | ...                              | ...                         | ...                                   |
| T               | d_T-NS_T           | d_T-NS_T'                             | NS_T                  | NS_T'                            | (d_T-NS_T)/d_T\*100%        | (d_T-NS_T')/d_T\*100%                 |
| **Total**       | **Σd_t-ΣNS_t**     | **Σd_t-ΣNS_t'**                       | **ΣNS_t**             | **ΣNS_t'**                       | **(Σd_t-ΣNS_t)/Σd_t\*100%** | **(Σd_t-ΣNS_t')/Σd_t\*100%**          |
| **Variación**   | **-**              | **Variación en demanda satisfecha**   | **-**                 | **ΣNS_t' - ΣNS_t**               | **-**                       | **Variación en % satisfacción**       |
| **% Variación** | **-**              | **% Variación en demanda satisfecha** | **-**                 | **(ΣNS_t' - ΣNS_t)/ΣNS_t\*100%** | **-**                       | **Puntos porcentuales de diferencia** |

#### Tabla 5: Comparación de costos

| Componente de costo                   | Base ($)             | Dotación mín. ($)       | Variación ($)                                 | % Variación                                                         |
| ------------------------------------- | -------------------- | ----------------------- | --------------------------------------------- | ------------------------------------------------------------------- |
| Personal contratado                   | cc*h*w₀\*T           | cc*h*w₀\*T              | 0                                             | 0%                                                                  |
| Personal por boleta                   | ct\*ΣW_t             | ct\*ΣW_t'               | ct\*(ΣW_t' - ΣW_t)                            | (ΣW_t' - ΣW_t)/ΣW_t × 100%                                          |
| Transformación a género               | g\*ΣY_t              | g\*ΣY_t'                | g\*(ΣY_t' - ΣY_t)                             | (ΣY_t' - ΣY_t)/ΣY_t × 100%                                          |
| Producción de prendas                 | n\*ΣZ_t              | n\*ΣZ_t'                | n\*(ΣZ_t' - ΣZ_t)                             | (ΣZ_t' - ΣZ_t)/ΣZ_t × 100%                                          |
| Almacenamiento                        | a\*Σ(IB_t+IM_t+IG_t) | a\*Σ(IB_t'+IM_t'+IG_t') | a\*[Σ(IB_t'+IM_t'+IG_t') - Σ(IB_t+IM_t+IG_t)] | [Σ(IB_t'+IM_t'+IG_t') - Σ(IB_t+IM_t+IG_t)]/Σ(IB_t+IM_t+IG_t) × 100% |
| Penalización por demanda insatisfecha | cp\*ΣNS_t            | cp\*ΣNS_t'              | cp\*(ΣNS_t' - ΣNS_t)                          | (ΣNS_t' - ΣNS_t)/ΣNS_t × 100%                                       |
| **Costo total**                       | **Z_base**           | **Z_dotmin**            | **Z_dotmin - Z_base**                         | **(Z_dotmin - Z_base)/Z_base × 100%**                               |

### Análisis de impacto

#### 1. Impacto en los costos totales

La política de dotación mínima de trabajadores probablemente resulte en:

- **Aumento en costos de personal**: Especialmente en periodos donde la demanda es baja y no se requeriría tantos trabajadores.
- **Posible sobreutilización de recursos**: En periodos con menor necesidad de producción, podría haber recursos humanos ociosos.
- **Posible reducción en penalizaciones**: La mayor disponibilidad de mano de obra podría permitir satisfacer mejor la demanda.

El impacto neto en los costos totales dependerá de si el incremento en costos de personal es compensado por mejoras en otros aspectos operativos.

#### 2. Impacto en los inventarios

La política podría generar:

- **Cambios en patrones de producción**: Al tener más capacidad de procesamiento disponible en algunos periodos, podría modificarse la planificación de producción.
- **Posible incremento en inventarios**: Si se procesa más material del necesario debido a la disponibilidad de mano de obra.
- **Cambios en la utilización de la capacidad de almacenamiento**: Dependiendo de cómo se ajusten los patrones de producción e inventario.

#### 3. Impacto en el cumplimiento de la demanda

La política de dotación mínima podría resultar en:

- **Mejor satisfacción de la demanda**: Al garantizar una capacidad mínima de procesamiento en cada periodo.
- **Reducción de penalizaciones por demanda insatisfecha**: Debido a la mayor disponibilidad de recursos para procesar material.
- **Mayor flexibilidad operativa**: Ante fluctuaciones inesperadas en la demanda o suministro.

## Análisis de trade-offs

La política de dotación mínima presenta importantes trade-offs:

1. **Costo vs. nivel de servicio**: Un aumento en los costos de personal podría justificarse si mejora significativamente el cumplimiento de demanda y reduce las penalizaciones.

2. **Eficiencia vs. seguridad operativa**: Aunque podría disminuir la eficiencia en uso de recursos humanos, proporciona una mayor seguridad operativa al garantizar una capacidad mínima de procesamiento.

3. **Flexibilidad vs. estructura**: La política reduce la flexibilidad en la contratación pero proporciona una estructura operativa más estable.

## Conclusiones preliminares

La implementación de una política de dotación mínima de tr trabajadores tendría los siguientes impactos generales:

1. **Efectos económicos**: Probablemente genere un incremento en los costos totales debido al aumento en costos de personal, aunque este incremento podría ser parcialmente compensado por mejoras en la satisfacción de demanda.

2. **Efectos operativos**: Podría modificar los patrones óptimos de producción e inventario, y proporcionar mayor estabilidad y capacidad de respuesta operativa.

3. **Efectos estratégicos**: La política representa un cambio de enfoque desde la optimización pura de costos hacia la garantía de un nivel mínimo de capacidad operativa.

La cuantificación exacta de estos impactos se determinará una vez resuelto el modelo actualizado con los parámetros específicos del problema.
