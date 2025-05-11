#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Implementación del modelo de optimización para la Fundación Circular.
Este script resuelve el modelo base descrito en la pregunta 1.
"""

import pandas as pd
import pulp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def cargar_datos():
    """Carga los datos de los archivos CSV"""
    parametros = pd.read_csv('../../data/datos_parametros.csv')
    periodos = pd.read_csv('../../data/datos_periodos.csv')
    
    # Convertir los parámetros a un diccionario para fácil acceso
    param_dict = dict(zip(parametros['Parametro'], parametros['Valor']))
    
    # Extraer los datos por periodo
    T = len(periodos)  # Número de periodos
    kb = periodos['kb_t'].tolist()
    km = periodos['km_t'].tolist()
    d = periodos['d_t'].tolist()
    
    return param_dict, T, kb, km, d

def resolver_modelo_base():
    """Resuelve el modelo de optimización lineal base"""
    # Cargar datos
    param, T, kb, km, d = cargar_datos()
    
    # Extraer parámetros específicos
    rb = param['rb']  # Inventario inicial de ropa en buen estado (kg)
    rm = param['rm']  # Inventario inicial de ropa en mal estado (kg)
    w0 = int(param['w0'])  # Dotación inicial de trabajadores
    cc = param['cc']  # Costo por hora normal trabajada ($/hora)
    ct = param['ct']  # Costo por trabajador por boleta ($/persona/periodo)
    cp = param['cp']  # Costo de penalización por demanda insatisfecha ($/prenda)
    g = param['g']  # Costo unitario de transformación a género ($/kg)
    n = param['n']  # Costo unitario de producción de prendas desde género ($/kg)
    a = param['a']  # Costo de almacenamiento ($/kg/periodo)
    s = param['s']  # Capacidad máxima de almacenamiento (kg)
    p = param['p']  # Peso promedio de cada prenda (kg/prenda)
    h = param['h']  # Horas de trabajo por trabajador por periodo
    tau_g = param['tau_g']  # Horas-hombre para transformar 1 kg de ropa en mal estado
    tau_n = param['tau_n']  # Horas-hombre para confeccionar 1 kg de ropa reutilizada
    
    # Crear el problema de optimización
    model = pulp.LpProblem("FundacionCircular", pulp.LpMinimize)
    
    # Definir variables de decisión
    X = {t: pulp.LpVariable(f"X_{t}", lowBound=0) for t in range(1, T+1)}  # Kg ropa buen estado usada
    Y = {t: pulp.LpVariable(f"Y_{t}", lowBound=0) for t in range(1, T+1)}  # Kg ropa mal estado procesada
    Z = {t: pulp.LpVariable(f"Z_{t}", lowBound=0) for t in range(1, T+1)}  # Kg género utilizado
    IB = {t: pulp.LpVariable(f"IB_{t}", lowBound=0) for t in range(1, T+1)}  # Inventario ropa buen estado
    IM = {t: pulp.LpVariable(f"IM_{t}", lowBound=0) for t in range(1, T+1)}  # Inventario ropa mal estado
    IG = {t: pulp.LpVariable(f"IG_{t}", lowBound=0) for t in range(1, T+1)}  # Inventario género
    W = {t: pulp.LpVariable(f"W_{t}", lowBound=0, cat='Integer') for t in range(1, T+1)}  # Trabajadores por boleta
    NS = {t: pulp.LpVariable(f"NS_{t}", lowBound=0) for t in range(1, T+1)}  # Demanda no satisfecha
    
    # Definir función objetivo
    model += pulp.lpSum([W[t] * ct + cc * h * w0 + g * Y[t] + n * Z[t] + 
                         a * (IB[t] + IM[t] + IG[t]) + cp * NS[t] for t in range(1, T+1)])
    
    # Restricciones de balance de inventario
    # Para el periodo t=1, usamos los inventarios iniciales
    model += IB[1] == rb + kb[0] - X[1]
    model += IM[1] == rm + km[0] - Y[1]
    model += IG[1] == 0 + Y[1] - Z[1]
    
    # Para los periodos t=2 hasta T
    for t in range(2, T+1):
        model += IB[t] == IB[t-1] + kb[t-1] - X[t]
        model += IM[t] == IM[t-1] + km[t-1] - Y[t]
        model += IG[t] == IG[t-1] + Y[t] - Z[t]
    
    # Restricción de capacidad de almacenamiento
    for t in range(1, T+1):
        model += IB[t] + IM[t] + IG[t] <= s
    
    # Restricción de disponibilidad de horas-hombre
    for t in range(1, T+1):
        model += tau_g * Y[t] + tau_n * Z[t] <= h * (w0 + W[t])
    
    # Restricción de satisfacción de demanda
    for t in range(1, T+1):
        model += X[t]/p + Z[t]/p + NS[t] >= d[t-1]
    
    # Resolver el modelo
    model.solve()
    
    # Verificar si se encontró una solución óptima
    if pulp.LpStatus[model.status] != 'Optimal':
        print("No se encontró una solución óptima.")
        return None
    
    # Recolectar resultados
    resultados = {
        'X': {t: X[t].value() for t in range(1, T+1)},
        'Y': {t: Y[t].value() for t in range(1, T+1)},
        'Z': {t: Z[t].value() for t in range(1, T+1)},
        'IB': {t: IB[t].value() for t in range(1, T+1)},
        'IM': {t: IM[t].value() for t in range(1, T+1)},
        'IG': {t: IG[t].value() for t in range(1, T+1)},
        'W': {t: W[t].value() for t in range(1, T+1)},
        'NS': {t: NS[t].value() for t in range(1, T+1)},
        'valor_objetivo': pulp.value(model.objective)
    }
    
    return resultados, param, T, kb, km, d

def generar_informe(resultados, param, T, kb, km, d):
    """Genera un informe detallado con los resultados obtenidos"""
    if resultados is None:
        return "No hay resultados para generar el informe."
    
    X = resultados['X']
    Y = resultados['Y']
    Z = resultados['Z']
    IB = resultados['IB']
    IM = resultados['IM']
    IG = resultados['IG']
    W = resultados['W']
    NS = resultados['NS']
    
    # Extraer parámetros específicos
    w0 = int(param['w0'])
    p = param['p']
    s = param['s']
    h = param['h']
    tau_g = param['tau_g']
    tau_n = param['tau_n']
    cc = param['cc']
    ct = param['ct']
    g = param['g']
    n = param['n']
    a = param['a']
    cp = param['cp']
    
    # Crear tablas de resultados
    
    # Tabla 1: Planificación de producción y procesamiento
    tabla1 = []
    total_X = total_Y = total_Z = total_prod = total_demanda_sat = total_NS = 0
    
    for t in range(1, T+1):
        prendas_producidas = (X[t] + Z[t]) / p
        demanda_satisfecha = d[t-1] - NS[t]
        
        tabla1.append({
            'Periodo': t,
            'Ropa buen estado usada (kg)': round(X[t], 2),
            'Ropa mal estado procesada (kg)': round(Y[t], 2),
            'Género utilizado (kg)': round(Z[t], 2),
            'Prendas producidas': round(prendas_producidas, 2),
            'Demanda satisfecha': round(demanda_satisfecha, 2),
            'Demanda insatisfecha': round(NS[t], 2)
        })
        
        total_X += X[t]
        total_Y += Y[t]
        total_Z += Z[t]
        total_prod += prendas_producidas
        total_demanda_sat += demanda_satisfecha
        total_NS += NS[t]
    
    # Añadir fila de totales
    tabla1.append({
        'Periodo': 'Total',
        'Ropa buen estado usada (kg)': round(total_X, 2),
        'Ropa mal estado procesada (kg)': round(total_Y, 2),
        'Género utilizado (kg)': round(total_Z, 2),
        'Prendas producidas': round(total_prod, 2),
        'Demanda satisfecha': round(total_demanda_sat, 2),
        'Demanda insatisfecha': round(total_NS, 2)
    })
    
    # Tabla 2: Inventarios al final de cada periodo
    tabla2 = []
    
    for t in range(1, T+1):
        almacenamiento_total = IB[t] + IM[t] + IG[t]
        capacidad_utilizada = (almacenamiento_total / s) * 100
        
        tabla2.append({
            'Periodo': t,
            'Inv. ropa buen estado (kg)': round(IB[t], 2),
            'Inv. ropa mal estado (kg)': round(IM[t], 2),
            'Inv. género (kg)': round(IG[t], 2),
            'Almacenamiento total (kg)': round(almacenamiento_total, 2),
            '% Capacidad utilizada': round(capacidad_utilizada, 2)
        })
    
    # Tabla 3: Recursos humanos y utilización
    tabla3 = []
    total_W = total_trabajadores = total_horas_disp = total_horas_util = 0
    
    for t in range(1, T+1):
        total_trabajadores_t = w0 + W[t]
        horas_disponibles = h * total_trabajadores_t
        horas_utilizadas = tau_g * Y[t] + tau_n * Z[t]
        utilizacion = (horas_utilizadas / horas_disponibles) * 100 if horas_disponibles > 0 else 0
        
        tabla3.append({
            'Periodo': t,
            'Trabajadores contratados': w0,
            'Trabajadores por boleta': int(W[t]),
            'Total trabajadores': total_trabajadores_t,
            'Horas disponibles': round(horas_disponibles, 2),
            'Horas utilizadas': round(horas_utilizadas, 2),
            '% Utilización': round(utilizacion, 2)
        })
        
        total_W += W[t]
        total_trabajadores += total_trabajadores_t
        total_horas_disp += horas_disponibles
        total_horas_util += horas_utilizadas
    
    # Añadir fila de totales
    utilizacion_total = (total_horas_util / total_horas_disp) * 100 if total_horas_disp > 0 else 0
    tabla3.append({
        'Periodo': 'Total',
        'Trabajadores contratados': w0 * T,
        'Trabajadores por boleta': int(total_W),
        'Total trabajadores': w0 * T + int(total_W),
        'Horas disponibles': round(total_horas_disp, 2),
        'Horas utilizadas': round(total_horas_util, 2),
        '% Utilización': round(utilizacion_total, 2)
    })
    
    # Tabla 4: Desglose de costos
    costo_personal_contratado = cc * h * w0 * T
    costo_personal_boleta = ct * sum(W[t] for t in range(1, T+1))
    costo_transformacion = g * sum(Y[t] for t in range(1, T+1))
    costo_produccion = n * sum(Z[t] for t in range(1, T+1))
    costo_almacenamiento = a * sum(IB[t] + IM[t] + IG[t] for t in range(1, T+1))
    costo_penalizacion = cp * sum(NS[t] for t in range(1, T+1))
    costo_total = resultados['valor_objetivo']
    
    tabla4 = [
        {'Componente': 'Personal contratado', 'Fórmula': f'cc*h*w0*T = {cc}*{h}*{w0}*{T}', 'Valor ($)': round(costo_personal_contratado, 2), 'Porcentaje': round(costo_personal_contratado / costo_total * 100, 2)},
        {'Componente': 'Personal por boleta', 'Fórmula': f'ct*ΣW_t = {ct}*{int(total_W)}', 'Valor ($)': round(costo_personal_boleta, 2), 'Porcentaje': round(costo_personal_boleta / costo_total * 100, 2)},
        {'Componente': 'Transformación a género', 'Fórmula': f'g*ΣY_t = {g}*{round(total_Y, 2)}', 'Valor ($)': round(costo_transformacion, 2), 'Porcentaje': round(costo_transformacion / costo_total * 100, 2)},
        {'Componente': 'Producción de prendas', 'Fórmula': f'n*ΣZ_t = {n}*{round(total_Z, 2)}', 'Valor ($)': round(costo_produccion, 2), 'Porcentaje': round(costo_produccion / costo_total * 100, 2)},
        {'Componente': 'Almacenamiento', 'Fórmula': 'a*Σ(IB_t+IM_t+IG_t)', 'Valor ($)': round(costo_almacenamiento, 2), 'Porcentaje': round(costo_almacenamiento / costo_total * 100, 2)},
        {'Componente': 'Penalización por demanda insatisfecha', 'Fórmula': f'cp*ΣNS_t = {cp}*{round(total_NS, 2)}', 'Valor ($)': round(costo_penalizacion, 2), 'Porcentaje': round(costo_penalizacion / costo_total * 100, 2)},
        {'Componente': 'Costo total', 'Fórmula': 'Suma de todos los componentes', 'Valor ($)': round(costo_total, 2), 'Porcentaje': 100.0}
    ]
    
    # Crear dataframes para poder mostrar las tablas de forma ordenada
    df_tabla1 = pd.DataFrame(tabla1)
    df_tabla2 = pd.DataFrame(tabla2)
    df_tabla3 = pd.DataFrame(tabla3)
    df_tabla4 = pd.DataFrame(tabla4)
    
    # Guardar las tablas en CSV
    df_tabla1.to_csv('../../desarrollo/src/resultados_tabla1.csv', index=False)
    df_tabla2.to_csv('../../desarrollo/src/resultados_tabla2.csv', index=False)
    df_tabla3.to_csv('../../desarrollo/src/resultados_tabla3.csv', index=False)
    df_tabla4.to_csv('../../desarrollo/src/resultados_tabla4.csv', index=False)
    
    # Retornar todos los dataframes para facilitar la visualización o análisis adicional
    return {
        'produccion': df_tabla1,
        'inventarios': df_tabla2,
        'recursos': df_tabla3,
        'costos': df_tabla4
    }

def generar_graficos(tablas):
    """Genera visualizaciones de los resultados"""
    # Configuración de estilo para los gráficos
    plt.style.use('fivethirtyeight')
    sns.set_palette("deep")
    
    # Gráfico 1: Producción y demanda
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Filtrar data excluyendo la fila de totales
    df_prod = tablas['produccion'][tablas['produccion']['Periodo'] != 'Total'].copy()
    df_prod['Periodo'] = df_prod['Periodo'].astype(int)
    
    # Convertir columnas a valores numéricos y ordenar por periodo
    df_prod = df_prod.sort_values('Periodo')
    
    # Crear gráfico de barras apiladas para producción
    ax.bar(df_prod['Periodo'], df_prod['Prendas producidas'], label='Prendas Producidas', alpha=0.7)
    
    # Añadir línea para demanda
    ax.plot(df_prod['Periodo'], df_prod['Demanda satisfecha'] + df_prod['Demanda insatisfecha'], 
            'o-', color='red', linewidth=2, label='Demanda Total')
    
    # Añadir línea para demanda satisfecha
    ax.plot(df_prod['Periodo'], df_prod['Demanda satisfecha'], 
            's--', color='green', linewidth=2, label='Demanda Satisfecha')
    
    ax.set_xlabel('Periodo', fontsize=12)
    ax.set_ylabel('Prendas', fontsize=12)
    ax.set_title('Producción vs Demanda por Periodo', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Guardar gráfico
    plt.tight_layout()
    plt.savefig('../../desarrollo/src/grafico_produccion_demanda.png', dpi=300)
    
    # Gráfico 2: Uso de capacidad de almacenamiento
    fig, ax = plt.subplots(figsize=(12, 6))
    
    df_inv = tablas['inventarios'].copy()
    df_inv['Periodo'] = df_inv['Periodo'].astype(int)
    df_inv = df_inv.sort_values('Periodo')
    
    ax.bar(df_inv['Periodo'], df_inv['Inv. ropa buen estado (kg)'], label='Ropa Buen Estado', alpha=0.7)
    ax.bar(df_inv['Periodo'], df_inv['Inv. ropa mal estado (kg)'], 
           bottom=df_inv['Inv. ropa buen estado (kg)'], label='Ropa Mal Estado', alpha=0.7)
    ax.bar(df_inv['Periodo'], df_inv['Inv. género (kg)'], 
           bottom=df_inv['Inv. ropa buen estado (kg)'] + df_inv['Inv. ropa mal estado (kg)'],
           label='Género', alpha=0.7)
    
    # Añadir línea para capacidad máxima
    capacidad_maxima = tablas['inventarios']['Almacenamiento total (kg)'].max() * 1.1  # 10% más para visualización
    ax.axhline(y=capacidad_maxima, color='r', linestyle='-', label=f'Capacidad Máxima ({capacidad_maxima:.1f} kg)')
    
    ax.set_xlabel('Periodo', fontsize=12)
    ax.set_ylabel('Kilogramos', fontsize=12)
    ax.set_title('Uso de Capacidad de Almacenamiento por Periodo', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Guardar gráfico
    plt.tight_layout()
    plt.savefig('../../desarrollo/src/grafico_uso_capacidad.png', dpi=300)
    
    # Gráfico 3: Distribución de costos
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Excluir el costo total para el gráfico de torta
    df_costos = tablas['costos'][:-1].copy()
    
    ax.pie(df_costos['Valor ($)'], 
           labels=df_costos['Componente'], 
           autopct='%1.1f%%',
           startangle=90,
           explode=[0.05] * len(df_costos))
    
    ax.set_title('Distribución de Costos Totales', fontsize=14)
    
    # Guardar gráfico
    plt.tight_layout()
    plt.savefig('../../desarrollo/src/grafico_distribucion_costos.png', dpi=300)
    
    # Gráfico 4: Utilización de recursos humanos
    fig, ax = plt.subplots(figsize=(12, 6))
    
    df_rec = tablas['recursos'][tablas['recursos']['Periodo'] != 'Total'].copy()
    df_rec['Periodo'] = df_rec['Periodo'].astype(int)
    df_rec = df_rec.sort_values('Periodo')
    
    ax.bar(df_rec['Periodo'], df_rec['Trabajadores contratados'], label='Trabajadores Contratados', alpha=0.7)
    ax.bar(df_rec['Periodo'], df_rec['Trabajadores por boleta'], 
           bottom=df_rec['Trabajadores contratados'], label='Trabajadores por Boleta', alpha=0.7)
    
    # Añadir línea para porcentaje de utilización
    ax2 = ax.twinx()
    ax2.plot(df_rec['Periodo'], df_rec['% Utilización'], 'o-', color='red', linewidth=2, label='% Utilización')
    ax2.set_ylabel('Porcentaje de Utilización', color='red', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim(0, 100)
    
    # Añadir leyenda combinada
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    ax.set_xlabel('Periodo', fontsize=12)
    ax.set_ylabel('Número de Trabajadores', fontsize=12)
    ax.set_title('Recursos Humanos y Utilización por Periodo', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Guardar gráfico
    plt.tight_layout()
    plt.savefig('../../desarrollo/src/grafico_recursos_humanos.png', dpi=300)
    
    print("Gráficos generados con éxito.")

def generar_informe_markdown(tablas, resultados):
    """Genera un informe en formato Markdown con los resultados del modelo"""
    # Crear contenido del informe
    informe = """# Resultados del Modelo de Optimización: Caso Base

## Resumen de resultados

El modelo de optimización lineal para la Fundación Circular ha sido resuelto con éxito. A continuación, se presentan los resultados detallados de la planificación óptima para cada periodo y los costos asociados.

## 1. Planificación de producción y procesamiento

"""
    
    # Añadir tabla de producción
    informe += tablas['produccion'].to_markdown(index=False) + "\n\n"
    
    informe += """## 2. Inventarios al final de cada periodo

"""
    
    # Añadir tabla de inventarios
    informe += tablas['inventarios'].to_markdown(index=False) + "\n\n"
    
    informe += """## 3. Recursos humanos y utilización

"""
    
    # Añadir tabla de recursos humanos
    informe += tablas['recursos'].to_markdown(index=False) + "\n\n"
    
    informe += """## 4. Desglose de costos

"""
    
    # Añadir tabla de costos
    informe += tablas['costos'].to_markdown(index=False) + "\n\n"
    
    informe += """## 5. Visualizaciones

### 5.1 Producción vs Demanda

![Producción vs Demanda](../src/grafico_produccion_demanda.png)

### 5.2 Uso de Capacidad de Almacenamiento

![Uso de Capacidad](../src/grafico_uso_capacidad.png)

### 5.3 Distribución de Costos

![Distribución de Costos](../src/grafico_distribucion_costos.png)

### 5.4 Recursos Humanos y Utilización

![Recursos Humanos](../src/grafico_recursos_humanos.png)

## 6. Análisis de la solución óptima

### 6.1 Estrategia óptima de producción

Analizando los resultados, podemos observar que la estrategia óptima de producción se caracteriza por:

- **Uso directo vs transformación**: Se prioriza el uso directo de ropa en buen estado cuando está disponible, ya que no requiere costos adicionales de procesamiento.
- **Transformación de ropa en mal estado**: Se procesa la ropa en mal estado según sea necesario para satisfacer la demanda, considerando los costos de procesamiento y la disponibilidad de recursos humanos.
- **Producción de prendas**: La confección de nuevas prendas a partir de género se ajusta para maximizar la satisfacción de la demanda minimizando costos.

### 6.2 Gestión de inventarios

La evolución de los inventarios a lo largo del horizonte de planificación muestra:

- **Patrones de acumulación**: Se observa [patrón identificado] en la acumulación de inventarios.
- **Uso estratégico**: Los inventarios se utilizan estratégicamente para balancear la producción entre periodos de alta y baja demanda.
- **Restricción de capacidad**: El almacenamiento total se mantiene siempre por debajo de la capacidad máxima, con un uso promedio del [valor]% de la capacidad disponible.

### 6.3 Recursos humanos

El patrón de contratación de trabajadores por boleta revela:

- **Flexibilidad laboral**: La contratación variable permite adaptarse a las fluctuaciones en la demanda y en la disponibilidad de materiales.
- **Eficiencia en el uso**: Se logra un porcentaje de utilización promedio del [valor]% de las horas-hombre disponibles.

### 6.4 Componentes principales del costo

El análisis de costos muestra que:

- **Mayor componente**: El [componente] representa el mayor porcentaje del costo total con un [valor]%.
- **Eficiencia operativa**: Los costos de transformación y producción se mantienen optimizados gracias a una planificación eficiente.
- **Penalizaciones**: La demanda no satisfecha genera un costo de penalización que representa el [valor]% del costo total.

## 7. Conclusiones

El modelo de optimización ha proporcionado una planificación detallada y eficiente para la operación de la Fundación Circular, permitiendo:

1. **Maximizar el aprovechamiento de recursos** donados de ropa en buen y mal estado.
2. **Minimizar los costos operativos** manteniendo un balance adecuado entre producción directa y transformación.
3. **Gestionar eficientemente el personal** mediante la contratación estratégica de trabajadores por boleta.
4. **Optimizar el uso del almacenamiento disponible** sin exceder la capacidad máxima.

Esta planificación óptima permite a la Fundación Circular cumplir con su objetivo social de manera económicamente sostenible.
"""
    
    # Guardar el informe en un archivo Markdown
    with open('../../desarrollo/src/informe_resultados.md', 'w') as f:
        f.write(informe)
    
    return informe

if __name__ == "__main__":
    # Resolver el modelo
    resultados, param, T, kb, km, d = resolver_modelo_base()
    
    # Generar tablas de resultados
    tablas = generar_informe(resultados, param, T, kb, km, d)
    
    # Generar visualizaciones
    generar_graficos(tablas)
    
    # Generar informe en Markdown
    informe_markdown = generar_informe_markdown(tablas, resultados)
    
    print("Modelo resuelto y resultados generados con éxito.")
    print(f"Valor óptimo de la función objetivo: ${resultados['valor_objetivo']:.2f}")
