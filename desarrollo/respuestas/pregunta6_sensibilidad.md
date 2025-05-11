# Pregunta 6: Análisis de Sensibilidad en la Demanda

## Planteamiento de la pregunta

> ¿Qué sucede si la demanda varía con respecto a lo pronosticado? Evalúe el impacto en la planificación si la demanda es un 80% y un 120% del valor original. Tabule y comente los resultados obtenidos. Redondee la demanda diaria a un número entero.

## Desarrollo de la respuesta

### Modificación del modelo

Para este análisis de sensibilidad, se evaluarán dos escenarios alternativos de demanda:

1. **Demanda reducida**: 80% de la demanda original
2. **Demanda incrementada**: 120% de la demanda original

Para cada periodo t, las demandas en estos escenarios serían:

- **Demanda original**: $d_t$ prendas
- **Demanda reducida**: $d_t^{80\%} = \lfloor 0.8 \times d_t \rfloor$ prendas (redondeado a entero)
- **Demanda incrementada**: $d_t^{120\%} = \lceil 1.2 \times d_t \rceil$ prendas (redondeado a entero)

Donde $\lfloor \cdot \rfloor$ representa redondeo hacia abajo y $\lceil \cdot \rceil$ representa redondeo hacia arriba.

La modificación en el código de implementación sería:

```python
# Demanda original
d_original = [100, 120, 130, 110, 100, 115]  # Ejemplo

# Escenario de demanda reducida (80%)
d_reducida = [int(0.8*d) for d in d_original]

# Escenario de demanda incrementada (120%)
import math
d_incrementada = [math.ceil(1.2*d) for d in d_original]

# Para resolver el modelo con demanda reducida
for t in range(T):
    # Actualizar la restricción de satisfacción de demanda
    problem += X[t]/p + Z[t]/p + NS[t] >= d_reducida[t]

# Para resolver el modelo con demanda incrementada
for t in range(T):
    # Actualizar la restricción de satisfacción de demanda
    problem += X[t]/p + Z[t]/p + NS[t] >= d_incrementada[t]
```

### Resultados comparativos

#### Tabla 1: Comparación de demanda por periodo

| Periodo   | Demanda original | Demanda reducida (80%) | Demanda incrementada (120%) |
| --------- | ---------------- | ---------------------- | --------------------------- |
| 1         | d₁               | ⌊0.8×d₁⌋               | ⌈1.2×d₁⌉                    |
| 2         | d₂               | ⌊0.8×d₂⌋               | ⌈1.2×d₂⌉                    |
| ...       | ...              | ...                    | ...                         |
| T         | d_T              | ⌊0.8×d_T⌋              | ⌈1.2×d_T⌉                   |
| **Total** | **Σd_t**         | **Σ⌊0.8×d_t⌋**         | **Σ⌈1.2×d_t⌉**              |

#### Tabla 2: Comparación de resultados operativos

| Indicador                                     | Demanda original           | Demanda reducida (80%)                 | Variación (%)                    | Demanda incrementada (120%)               | Variación (%)                     |
| --------------------------------------------- | -------------------------- | -------------------------------------- | -------------------------------- | ----------------------------------------- | --------------------------------- |
| **Procesamiento y producción**                |                            |                                        |                                  |                                           |                                   |
| Ropa en buen estado utilizada (kg)            | ΣX_t                       | ΣX_t^80%                               | (ΣX_t^80% - ΣX_t)/ΣX_t × 100%    | ΣX_t^120%                                 | (ΣX_t^120% - ΣX_t)/ΣX_t × 100%    |
| Ropa en mal estado procesada (kg)             | ΣY_t                       | ΣY_t^80%                               | (ΣY_t^80% - ΣY_t)/ΣY_t × 100%    | ΣY_t^120%                                 | (ΣY_t^120% - ΣY_t)/ΣY_t × 100%    |
| Género utilizado (kg)                         | ΣZ_t                       | ΣZ_t^80%                               | (ΣZ_t^80% - ΣZ_t)/ΣZ_t × 100%    | ΣZ_t^120%                                 | (ΣZ_t^120% - ΣZ_t)/ΣZ_t × 100%    |
| **Recursos humanos**                          |                            |                                        |                                  |                                           |                                   |
| Trabajadores por boleta (total)               | ΣW_t                       | ΣW_t^80%                               | (ΣW_t^80% - ΣW_t)/ΣW_t × 100%    | ΣW_t^120%                                 | (ΣW_t^120% - ΣW_t)/ΣW_t × 100%    |
| Utilización de horas-hombre (promedio)        | Promedio                   | Promedio^80%                           | Diferencia                       | Promedio^120%                             | Diferencia                        |
| **Demanda**                                   |                            |                                        |                                  |                                           |                                   |
| Demanda total (prendas)                       | Σd_t                       | Σd_t^80%                               | -20%                             | Σd_t^120%                                 | +20%                              |
| Demanda satisfecha (prendas)                  | Σd_t - ΣNS_t               | Σd_t^80% - ΣNS_t^80%                   | Variación                        | Σd_t^120% - ΣNS_t^120%                    | Variación                         |
| Demanda no satisfecha (prendas)               | ΣNS_t                      | ΣNS_t^80%                              | (ΣNS_t^80% - ΣNS_t)/ΣNS_t × 100% | ΣNS_t^120%                                | (ΣNS_t^120% - ΣNS_t)/ΣNS_t × 100% |
| % Satisfacción global                         | (Σd_t - ΣNS_t)/Σd_t × 100% | (Σd_t^80% - ΣNS_t^80%)/Σd_t^80% × 100% | Diferencia en puntos %           | (Σd_t^120% - ΣNS_t^120%)/Σd_t^120% × 100% | Diferencia en puntos %            |
| **Inventarios**                               |                            |                                        |                                  |                                           |                                   |
| Inventario promedio total (kg)                | Σ(IB_t+IM_t+IG_t)/T        | Σ(IB_t^80%+IM_t^80%+IG_t^80%)/T        | Variación %                      | Σ(IB_t^120%+IM_t^120%+IG_t^120%)/T        | Variación %                       |
| Utilización promedio capacidad almacenamiento | Promedio %                 | Promedio^80% %                         | Diferencia en puntos %           | Promedio^120% %                           | Diferencia en puntos %            |

#### Tabla 3: Comparación de costos

| Componente de costo                   | Demanda original ($)          | Demanda reducida (80%) ($)       | Variación (%)                              | Demanda incrementada (120%) ($)     | Variación (%)                               |
| ------------------------------------- | ----------------------------- | -------------------------------- | ------------------------------------------ | ----------------------------------- | ------------------------------------------- |
| Personal contratado                   | cc*h*w₀\*T                    | cc*h*w₀\*T                       | 0%                                         | cc*h*w₀\*T                          | 0%                                          |
| Personal por boleta                   | ct\*ΣW_t                      | ct\*ΣW_t^80%                     | (ΣW_t^80% - ΣW_t)/ΣW_t × 100%              | ct\*ΣW_t^120%                       | (ΣW_t^120% - ΣW_t)/ΣW_t × 100%              |
| Transformación a género               | g\*ΣY_t                       | g\*ΣY_t^80%                      | (ΣY_t^80% - ΣY_t)/ΣY_t × 100%              | g\*ΣY_t^120%                        | (ΣY_t^120% - ΣY_t)/ΣY_t × 100%              |
| Producción de prendas                 | n\*ΣZ_t                       | n\*ΣZ_t^80%                      | (ΣZ_t^80% - ΣZ_t)/ΣZ_t × 100%              | n\*ΣZ_t^120%                        | (ΣZ_t^120% - ΣZ_t)/ΣZ_t × 100%              |
| Almacenamiento                        | a\*Σ(IB_t+IM_t+IG_t)          | a\*Σ(IB_t^80%+IM_t^80%+IG_t^80%) | Variación %                                | a\*Σ(IB_t^120%+IM_t^120%+IG_t^120%) | Variación %                                 |
| Penalización por demanda insatisfecha | cp\*ΣNS_t                     | cp\*ΣNS_t^80%                    | (ΣNS_t^80% - ΣNS_t)/ΣNS_t × 100%           | cp\*ΣNS_t^120%                      | (ΣNS_t^120% - ΣNS_t)/ΣNS_t × 100%           |
| **Costo total**                       | **Z_original**                | **Z_80%**                        | **(Z_80% - Z_original)/Z_original × 100%** | **Z_120%**                          | **(Z_120% - Z_original)/Z_original × 100%** |
| **Costo por prenda satisfecha**       | **Z_original/(Σd_t - ΣNS_t)** | **Z_80%/(Σd_t^80% - ΣNS_t^80%)** | **Variación %**                            | **Z_120%/(Σd_t^120% - ΣNS_t^120%)** | **Variación %**                             |

### Análisis de impacto

#### 1. Impacto de una demanda reducida (80%)

La reducción de la demanda en un 20% probablemente resulte en:

- **Menor necesidad de recursos**: Reducción en el procesamiento de ropa en mal estado, utilización de género y contratación de trabajadores por boleta.
- **Menor costo total**: Debido a la reducción en costos de procesamiento, personal y potencialmente menos penalizaciones.
- **Mayor tasa de satisfacción de demanda**: Al tener que satisfacer menos demanda con los mismos recursos iniciales.
- **Posible aumento relativo en inventarios**: Al procesar menos material del que ingresa al sistema.

#### 2. Impacto de una demanda incrementada (120%)

El incremento de la demanda en un 20% probablemente resulte en:

- **Mayor necesidad de recursos**: Incremento en el procesamiento de ropa en mal estado, utilización de género y contratación de trabajadores por boleta.
- **Mayor costo total**: Debido al aumento en costos de procesamiento, personal y potencialmente más penalizaciones.
- **Menor tasa de satisfacción de demanda**: Al tener que satisfacer más demanda con restricciones en los recursos disponibles.
- **Posible reducción relativa en inventarios**: Al procesar más material para satisfacer la mayor demanda.

#### 3. Elasticidad de costos respecto a la demanda

Es interesante analizar cómo responden los costos ante cambios en la demanda:

- **Elasticidad de costos**: Si una reducción del 20% en la demanda genera una reducción menor al 20% en los costos totales, el sistema presenta economías de escala.
- **Economías de escala**: Se evaluará si el costo por prenda satisfecha es menor cuando la demanda es mayor, lo que indicaría la presencia de economías de escala.

#### 4. Robustez del modelo

El análisis de sensibilidad permite evaluar la robustez del modelo ante variaciones en la demanda:

- **Capacidad de absorción**: Capacidad del sistema para absorber incrementos en la demanda sin grandes aumentos en costos o reducciones en la satisfacción.
- **Flexibilidad operativa**: Capacidad de ajustar recursos y operaciones ante fluctuaciones en la demanda.

## Conclusiones preliminares

El análisis de sensibilidad en la demanda permite extraer las siguientes conclusiones:

1. **Escenario de demanda reducida (80%)**: Probablemente resulte en menores costos totales, mejor satisfacción de la demanda y menor utilización de recursos. Sin embargo, el costo por prenda satisfecha podría ser mayor debido a los costos fijos del sistema.

2. **Escenario de demanda incrementada (120%)**: Probablemente resulte en mayores costos totales, menor satisfacción de la demanda y mayor utilización de recursos. Sin embargo, el costo por prenda satisfecha podría ser menor si existen economías de escala.

3. **Implicaciones para la planificación**: Los resultados sugieren que la fundación debería:
   - Desarrollar mecanismos de flexibilidad para ajustar sus operaciones ante variaciones en la demanda.
   - Considerar mantener cierta capacidad de reserva para absorber potenciales incrementos en la demanda.
   - Evaluar estrategias para reducir los costos fijos y mejorar la eficiencia operativa ante posibles reducciones en la demanda.

La cuantificación exacta de estos impactos se determinará una vez resueltos los modelos actualizados con los diferentes escenarios de demanda.
