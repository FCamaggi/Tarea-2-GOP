#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Implementación del modelo de optimización para la Fundación Circular.
Este script resuelve el modelo para la pregunta 2.
"""

import pandas as pd
import pulp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patheffects
import seaborn as sns
import os
from pathlib import Path

# Asegurar que las rutas sean correctas independientemente de donde se ejecute el script
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "desarrollo" / "src" / "pregunta2"
RESPUESTAS_DIR = BASE_DIR / "desarrollo" / "respuestas"

def cargar_datos():
    """Carga los datos de los archivos CSV"""
    parametros = pd.read_csv(DATA_DIR / 'datos_parametros.csv')
    periodos = pd.read_csv(DATA_DIR / 'datos_periodos.csv')
    
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
        model += (1/p)*X[t] + (1/p)*Z[t] + NS[t] >= d[t-1]
    
    # Resolver el modelo
    model.solve(pulp.PULP_CBC_CMD(msg=False))
    
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
        'W': {t: int(W[t].value()) for t in range(1, T+1)},
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
            'Trabajadores por boleta': W[t],
            'Total trabajadores': total_trabajadores_t,
            'Horas disponibles': round(horas_disponibles, 2),
            'Horas utilizadas': round(horas_utilizadas, 2),
            '% Utilización': round(utilizacion, 2)
        })
        
        total_W += W[t]
        total_trabajadores += total_trabajadores_t
        total_horas_disp += horas_disponibles
        total_horas_util += horas_utilizadas
    
    # Añadir fila de totales/promedio
    utilizacion_total = (total_horas_util / total_horas_disp) * 100 if total_horas_disp > 0 else 0
    tabla3.append({
        'Periodo': 'Total',
        'Trabajadores contratados': w0 * T,
        'Trabajadores por boleta': total_W,
        'Total trabajadores': w0 * T + total_W,
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
        {'Componente': 'Personal por boleta', 'Fórmula': f'ct*ΣW_t = {ct}*{total_W}', 'Valor ($)': round(costo_personal_boleta, 2), 'Porcentaje': round(costo_personal_boleta / costo_total * 100, 2)},
        {'Componente': 'Transformación a género', 'Fórmula': f'g*ΣY_t = {g}*{round(total_Y, 2)}', 'Valor ($)': round(costo_transformacion, 2), 'Porcentaje': round(costo_transformacion / costo_total * 100, 2)},
        {'Componente': 'Producción de prendas', 'Fórmula': f'n*ΣZ_t = {n}*{round(total_Z, 2)}', 'Valor ($)': round(costo_produccion, 2), 'Porcentaje': round(costo_produccion / costo_total * 100, 2)},
        {'Componente': 'Almacenamiento', 'Fórmula': f'a*Σ(IB_t+IM_t+IG_t) = {a}*{round(sum(IB[t] + IM[t] + IG[t] for t in range(1, T+1)), 2)}', 'Valor ($)': round(costo_almacenamiento, 2), 'Porcentaje': round(costo_almacenamiento / costo_total * 100, 2)},
        {'Componente': 'Penalización por demanda insatisfecha', 'Fórmula': f'cp*ΣNS_t = {cp}*{round(total_NS, 2)}', 'Valor ($)': round(costo_penalizacion, 2), 'Porcentaje': round(costo_penalizacion / costo_total * 100, 2)},
        {'Componente': 'Costo total', 'Fórmula': 'Suma de todos los componentes', 'Valor ($)': round(costo_total, 2), 'Porcentaje': 100.0}
    ]
    
    # Crear dataframes para poder mostrar las tablas de forma ordenada
    df_tabla1 = pd.DataFrame(tabla1)
    df_tabla2 = pd.DataFrame(tabla2)
    df_tabla3 = pd.DataFrame(tabla3)
    df_tabla4 = pd.DataFrame(tabla4)
    
    # Guardar las tablas en CSV para referencia
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df_tabla1.to_csv(OUTPUT_DIR / 'resultados_tabla1_p2.csv', index=False)
    df_tabla2.to_csv(OUTPUT_DIR / 'resultados_tabla2_p2.csv', index=False)
    df_tabla3.to_csv(OUTPUT_DIR / 'resultados_tabla3_p2.csv', index=False)
    df_tabla4.to_csv(OUTPUT_DIR / 'resultados_tabla4_p2.csv', index=False)
    
    # Retornar todos los dataframes para facilitar la visualización o análisis adicional
    return {
        'produccion': df_tabla1,
        'inventarios': df_tabla2,
        'recursos': df_tabla3,
        'costos': df_tabla4
    }

def generar_graficos(tablas):
    """Genera visualizaciones mejoradas de los resultados"""
    # Configuración de estilo para los gráficos
    sns.set_theme(style="darkgrid")  # Configurar el tema de seaborn
    
    # Definir una paleta de colores personalizada más profesional
    paleta_colores = {
        'azul_principal': '#2c3e50',
        'verde_exito': '#27ae60',
        'rojo_error': '#c0392b',
        'naranja_alerta': '#e67e22',
        'azul_claro': '#3498db',
        'gris_neutro': '#95a5a6'
    }
    
    # Configurar el estilo general de las fuentes
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    
    # Cargar parámetros del modelo
    parametros = pd.read_csv(DATA_DIR / 'datos_parametros.csv')
    param = dict(zip(parametros['Parametro'], parametros['Valor']))
    capacidad_maxima = param['s']  # Capacidad máxima de almacenamiento
    
    # =====================================================
    # 1. Gráfico mejorado de Producción vs Demanda
    # =====================================================
    fig, ax = plt.subplots(figsize=(12, 7), facecolor='white')
    ax.set_facecolor('#f8f9fa')
    
    # Filtrar data excluyendo la fila de totales
    df_prod = tablas['produccion'][tablas['produccion']['Periodo'] != 'Total'].copy()
    df_prod['Periodo'] = df_prod['Periodo'].astype(int)
    df_prod = df_prod.sort_values('Periodo')
    
    # Obtener valores numéricos
    periodos = df_prod['Periodo']
    demanda_total = df_prod['Demanda satisfecha'] + df_prod['Demanda insatisfecha']
    demanda_satisfecha = df_prod['Demanda satisfecha']
    demanda_insatisfecha = df_prod['Demanda insatisfecha']
    prendas_producidas = df_prod['Prendas producidas']
    
    # Crear gráfico apilado para la demanda
    ax.bar(periodos, demanda_satisfecha, label='Demanda Satisfecha', alpha=0.8,
           color=paleta_colores['verde_exito'])
    ax.bar(periodos, demanda_insatisfecha, bottom=demanda_satisfecha,
           label='Demanda Insatisfecha', alpha=0.8, color=paleta_colores['rojo_error'])
    
    # Añadir línea para producción
    ax.plot(periodos, prendas_producidas, 'o-', color=paleta_colores['azul_principal'],
            linewidth=2.5, label='Prendas Producidas', markersize=8,
            path_effects=[patheffects.withStroke(linewidth=4, foreground='white')])
    
    # Añadir anotaciones con porcentajes de demanda satisfecha
    for i, periodo in enumerate(periodos):
        porcentaje = (demanda_satisfecha.iloc[i] / demanda_total.iloc[i]) * 100
        if porcentaje > 10:  # Solo mostrar etiqueta si hay suficiente espacio
            ax.annotate(f'{porcentaje:.1f}%\n({demanda_satisfecha.iloc[i]:.0f})',
                        xy=(periodo, demanda_satisfecha.iloc[i] / 2),
                        ha='center', va='center',
                        color='white', fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.5', fc=(0,0,0,0.3), ec='none'))
    
    # Añadir etiquetas de valor encima de cada barra
    for i, periodo in enumerate(periodos):
        ax.annotate(f'Total: {demanda_total.iloc[i]:.0f}', 
                    xy=(periodo, demanda_total.iloc[i] + 20), 
                    ha='center', color='#333333')
    
    ax.set_xlabel('Periodo de Planificación', fontsize=14, fontweight='bold')
    ax.set_ylabel('Cantidad de Prendas', fontsize=14, fontweight='bold')
    ax.set_title('Análisis de Producción y Satisfacción de Demanda\npor Periodo de Planificación',
                 fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper left', fontsize=12)
    
    # Personalizar ejes
    ax.set_xticks(periodos)
    ax.tick_params(axis='both', labelsize=12)
    
    # Añadir resumen en texto
    props = dict(boxstyle='round', facecolor='white', alpha=0.7)
    demanda_total_suma = tablas['produccion'][tablas['produccion']['Periodo'] == 'Total']['Demanda satisfecha'].iloc[0] + tablas['produccion'][tablas['produccion']['Periodo'] == 'Total']['Demanda insatisfecha'].iloc[0]
    demanda_sat_suma = tablas['produccion'][tablas['produccion']['Periodo'] == 'Total']['Demanda satisfecha'].iloc[0]
    porcentaje_total = (demanda_sat_suma / demanda_total_suma) * 100
    
    textstr = f'Demanda total: {demanda_total_suma:.0f} prendas\n'
    textstr += f'Demanda satisfecha: {demanda_sat_suma:.0f} prendas ({porcentaje_total:.1f}%)\n'
    textstr += f'Demanda insatisfecha: {demanda_total_suma - demanda_sat_suma:.0f} prendas ({100-porcentaje_total:.1f}%)'
    
    ax.text(0.98, 0.98, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', horizontalalignment='right', bbox=props)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'grafico_produccion_demanda_p2.png', dpi=300)
    
    # =====================================================
    # 2. Gráfico mejorado de Uso de Capacidad de Almacenamiento
    # =====================================================
    fig, ax = plt.subplots(figsize=(12, 7), facecolor='white')
    ax.set_facecolor('#f8f9fa')
    
    # Agregar líneas de referencia horizontales sutiles
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    
    # Filtrar y ordenar datos
    df_inv = tablas['inventarios'].copy()
    df_inv['Periodo'] = df_inv['Periodo'].astype(int)
    df_inv = df_inv.sort_values('Periodo')
    
    # Componentes de inventario
    inv_buen_estado = df_inv['Inv. ropa buen estado (kg)']
    inv_mal_estado = df_inv['Inv. ropa mal estado (kg)']
    inv_genero = df_inv['Inv. género (kg)']
    almacenamiento_total = df_inv['Almacenamiento total (kg)']
    capacidad_utilizada = df_inv['% Capacidad utilizada']
    
    # Crear gráfico de barras apiladas
    width = 0.5  # Ancho de las barras
    p1 = ax.bar(periodos, inv_buen_estado, width,
                label='Ropa en Buen Estado', color=paleta_colores['azul_claro'])
    p2 = ax.bar(periodos, inv_mal_estado, width,
                bottom=inv_buen_estado, label='Ropa en Mal Estado',
                color=paleta_colores['naranja_alerta'])
    p3 = ax.bar(periodos, inv_genero, width,
                bottom=inv_buen_estado+inv_mal_estado, label='Género Procesado',
                color=paleta_colores['verde_exito'])
    
    # Añadir línea para capacidad máxima
    ax.axhline(y=capacidad_maxima, color='#e74c3c', linestyle='--', 
              linewidth=2, label=f'Capacidad Máxima ({capacidad_maxima:.0f} kg)')
    
    # Añadir etiquetas para el porcentaje de utilización
    for i, periodo in enumerate(periodos):
        # Formato condicional para las etiquetas según el nivel de utilización
        color = '#2ecc71' if capacidad_utilizada.iloc[i] < 70 else '#e74c3c'
        ax.annotate(f'{capacidad_utilizada.iloc[i]:.1f}%\n({almacenamiento_total.iloc[i]:.0f} kg)',
                    xy=(periodo, almacenamiento_total.iloc[i] + 20),
                    ha='center', va='bottom', fontweight='bold',
                   color=color,
                   bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='none', alpha=0.8))
    
    ax.set_xlabel('Periodo de Planificación', fontsize=14, fontweight='bold')
    ax.set_ylabel('Nivel de Inventario (kg)', fontsize=14, fontweight='bold')
    ax.set_title('Evolución y Composición de Inventarios\ncon Porcentaje de Utilización de Capacidad',
                 fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper right', fontsize=12)
    
    # Personalizar ejes
    ax.set_xticks(periodos)
    ax.tick_params(axis='both', labelsize=12)
    
    # Añadir un segundo eje y para mostrar el % de utilización
    ax2 = ax.twinx()
    ax2.set_ylabel('% de Capacidad Utilizada', color='#e74c3c', fontsize=14, fontweight='bold')
    ax2.plot(periodos, capacidad_utilizada, 'o-', color='#e74c3c', linewidth=2, markersize=8)
    ax2.tick_params(axis='y', labelcolor='#e74c3c')
    ax2.set_ylim(0, 100)  # Fijar el límite del eje y secundario
    
    # Añadir resumen en texto
    props = dict(boxstyle='round', facecolor='white', alpha=0.7)
    textstr = f'Capacidad máxima: {capacidad_maxima:.0f} kg\n'
    textstr += f'Utilización promedio: {capacidad_utilizada.mean():.2f}%\n'
    textstr += f'Máxima utilización: {capacidad_utilizada.max():.2f}% (Periodo {capacidad_utilizada.idxmax()+1})'
    
    ax.text(0.98, 0.02, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='bottom', horizontalalignment='right', bbox=props)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'grafico_uso_capacidad_p2.png', dpi=300)
    
    # =====================================================
    # 3. Gráfico mejorado de Distribución de Costos
    # =====================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8), facecolor='white')
    ax1.set_facecolor('#f8f9fa')
    # Cambio de fondo para el gráfico de barras para mejorar visibilidad
    ax2.set_facecolor('#e0e0e0')  # Gris más oscuro para mejor contraste
    
    # Excluir el costo total para el gráfico de torta
    df_costos = tablas['costos'][:-1].copy()
    
    # Agrupar componentes pequeños en "Otros" para mejor visualización
    umbral = 5.0  # Agrupar componentes que representan menos del 5%
    componentes_pequeños = df_costos[df_costos['Porcentaje'] < umbral]
    componentes_grandes = df_costos[df_costos['Porcentaje'] >= umbral]
    
    if len(componentes_pequeños) > 0:
        otros = pd.DataFrame({
            'Componente': ['Otros componentes'],
            'Fórmula': ['Suma de componentes menores'],
            'Valor ($)': [componentes_pequeños['Valor ($)'].sum()],
            'Porcentaje': [componentes_pequeños['Porcentaje'].sum()]
        })
        df_costos_agrupados = pd.concat([componentes_grandes, otros])
    else:
        df_costos_agrupados = componentes_grandes.copy()
    
    # Ordenar por porcentaje descendente
    df_costos_agrupados = df_costos_agrupados.sort_values('Porcentaje', ascending=False)
    
    # Colores para el gráfico de torta
    colors = plt.cm.viridis(np.linspace(0, 1, len(df_costos_agrupados)))
    
    # Gráfico de torta (izquierda)
    explode = [0.05 if i == 0 else 0 for i in range(len(df_costos_agrupados))]  # Destacar el componente mayor
    wedges, texts, autotexts = ax1.pie(
        df_costos_agrupados['Porcentaje'], 
        autopct='%1.1f%%',
        explode=explode,
        shadow=True,
        startangle=90,
        colors=colors
    )
    
    # Hacer las etiquetas más legibles
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(12)
        autotext.set_fontweight('bold')
    
    ax1.set_title('Distribución Porcentual\nde Componentes de Costo',
                  fontsize=16, fontweight='bold', pad=20)
    ax1.axis('equal')  # Para asegurar que el círculo es perfecto
    
    # Gráfico de barras horizontales (derecha) - Muestra valores absolutos
    y_pos = np.arange(len(df_costos_agrupados))
    # Crear barras horizontales con degradado
    bars = ax2.barh(y_pos, df_costos_agrupados['Valor ($)'],
                    color=colors, edgecolor='black', linewidth=1,
                    alpha=0.8)
    
    # Agregar línea de tendencia acumulativa
    valores_acumulados = df_costos_agrupados['Valor ($)'].cumsum()
    ax2.plot(valores_acumulados, y_pos, 'o--',
             color=paleta_colores['azul_principal'],
             alpha=0.6, label='Acumulado')
    
    # Añadir etiquetas con valores
    for i, bar in enumerate(bars):
        valor = df_costos_agrupados['Valor ($)'].iloc[i]
        # Formato monetario mejorado
        if valor >= 1e9:  # Billones
            valor_str = f'${valor/1e9:.1f}B ({df_costos_agrupados["Porcentaje"].iloc[i]:.1f}%)'
        elif valor >= 1e6:  # Millones
            valor_str = f'${valor/1e6:.1f}M ({df_costos_agrupados["Porcentaje"].iloc[i]:.1f}%)'
        elif valor >= 1e3:  # Miles
            valor_str = f'${valor/1e3:.0f}K ({df_costos_agrupados["Porcentaje"].iloc[i]:.1f}%)'
        else:
            valor_str = f'${valor:.0f} ({df_costos_agrupados["Porcentaje"].iloc[i]:.1f}%)'
        
        # Añadir texto con efecto de contorno para mejor visibilidad
        text = ax2.text(bar.get_width() * 0.02, bar.get_y() + bar.get_height()/2, 
                 valor_str, va='center', color='black', fontweight='bold', fontsize=12)
        # Agregar contorno al texto para mejor visibilidad
        text.set_path_effects([patheffects.withStroke(linewidth=3, foreground='white')])
    
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(df_costos_agrupados['Componente'])
    ax2.set_xlabel('Costo ($)', fontsize=14, fontweight='bold')
    ax2.set_title('Desglose de Costos por Componente\n(Valores Absolutos)',
                  fontsize=16, fontweight='bold', pad=20)
    ax2.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Añadir costo total como anotación
    costo_total = tablas['costos'].iloc[-1]['Valor ($)']
    if costo_total >= 1e6:
        costo_str = f'${costo_total/1e6:.2f} millones (100%)'
    elif costo_total >= 1e3:
        costo_str = f'${costo_total/1e3:.0f}K (100%)'
    else:
        costo_str = f'${costo_total:,.2f} (100%)'
    
    fig.text(0.5, 0.01, f'Costo Total: {costo_str}', ha='center', 
             fontsize=14, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Ajustar para el texto del costo total
    plt.savefig(OUTPUT_DIR / 'grafico_distribucion_costos_p2.png', dpi=300)
    
    # =====================================================
    # 4. Gráfico mejorado de Recursos Humanos
    # =====================================================
    fig, ax = plt.subplots(figsize=(12, 7), facecolor='white')
    ax.set_facecolor('#f8f9fa')
    
    # Agregar líneas de referencia horizontales sutiles
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    
    # Filtrar data excluyendo la fila de totales
    df_rec = tablas['recursos'][tablas['recursos']['Periodo'] != 'Total'].copy()
    df_rec['Periodo'] = df_rec['Periodo'].astype(int)
    df_rec = df_rec.sort_values('Periodo')
    
    # Obtener valores
    trabajadores_contratados = df_rec['Trabajadores contratados']
    trabajadores_boleta = df_rec['Trabajadores por boleta']
    total_trabajadores = df_rec['Total trabajadores']
    horas_disponibles = df_rec['Horas disponibles']
    horas_utilizadas = df_rec['Horas utilizadas']
    
    # Crear gráfico de barras apiladas
    width = 0.5
    p1 = ax.bar(periodos, trabajadores_contratados, width,
                label='Trabajadores Permanentes', color=paleta_colores['azul_principal'])
    p2 = ax.bar(periodos, trabajadores_boleta, width, bottom=trabajadores_contratados,
                label='Trabajadores Temporales (Boleta)', color=paleta_colores['naranja_alerta'])
    
    # Añadir etiquetas con el total de trabajadores
    for i, periodo in enumerate(periodos):
        total = int(total_trabajadores.iloc[i])
        perm = int(trabajadores_contratados.iloc[i])
        temp = int(trabajadores_boleta.iloc[i])
        
        ax.annotate(f'Total: {total}\n(Perm: {perm}, Temp: {temp})',
                    xy=(periodo, total_trabajadores.iloc[i] + 0.5),
                    ha='center', va='bottom', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='none', alpha=0.8))
    
    # Añadir horas como anotaciones
    for i, periodo in enumerate(periodos):
        ax.annotate(f'{horas_utilizadas.iloc[i]:.0f} horas', 
                   xy=(periodo, total_trabajadores.iloc[i]/2), 
                   ha='center', va='center', color='white', fontweight='bold')
    
    ax.set_xlabel('Periodo de Planificación', fontsize=14, fontweight='bold')
    ax.set_ylabel('Dotación de Personal', fontsize=14, fontweight='bold')
    ax.set_title('Composición y Distribución de la Fuerza Laboral\npor Periodo de Planificación',
                 fontsize=16, fontweight='bold', pad=20)
    
    # Configuración de ejes
    ax.set_xticks(periodos)
    ax.tick_params(axis='both', labelsize=12)
    
    # Leyenda
    ax.legend(loc='upper right', fontsize=12)
    
    # Añadir resumen en texto
    props = dict(boxstyle='round', facecolor='white', alpha=0.7)
    
    total_contratados = tablas['recursos'][tablas['recursos']['Periodo'] == 'Total']['Trabajadores contratados'].iloc[0]
    total_boleta = tablas['recursos'][tablas['recursos']['Periodo'] == 'Total']['Trabajadores por boleta'].iloc[0]
    total_horas = tablas['recursos'][tablas['recursos']['Periodo'] == 'Total']['Horas utilizadas'].iloc[0]
    
    textstr = f'Total trabajadores contratados: {total_contratados}\n'
    textstr += f'Total trabajadores por boleta: {total_boleta}\n'
    textstr += f'Total horas trabajadas: {total_horas:.0f} horas'
    
    ax.text(0.98, 0.02, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='bottom', horizontalalignment='right', bbox=props)
    
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'grafico_recursos_humanos_p2.png', dpi=300)
    
    print("Gráficos mejorados generados con éxito.")

def generar_informe_markdown(tablas):
    """Genera un informe en formato Markdown con los resultados del modelo"""
    # Obtener los datos necesarios para generar el análisis
    df_prod = tablas['produccion']
    df_inv = tablas['inventarios']
    df_rec = tablas['recursos']
    df_costos = tablas['costos']
    
    # Calcular algunos KPIs para el análisis
    prod_total = float(df_prod.loc[df_prod['Periodo'] == 'Total', 'Prendas producidas'].iloc[0])
    demanda_sat_total = float(df_prod.loc[df_prod['Periodo'] == 'Total', 'Demanda satisfecha'].iloc[0])
    demanda_total = demanda_sat_total + float(df_prod.loc[df_prod['Periodo'] == 'Total', 'Demanda insatisfecha'].iloc[0])
    porcentaje_demanda_sat = (demanda_sat_total / demanda_total) * 100 if demanda_total > 0 else 0
    
    utilizacion_promedio = float(df_rec.loc[df_rec['Periodo'] == 'Total', '% Utilización'].iloc[0])
    
    # Obtener el componente de mayor costo
    df_costos_sin_total = df_costos[:-1]  # Excluir el total
    componente_mayor_costo = df_costos_sin_total.loc[df_costos_sin_total['Valor ($)'].idxmax()]
    
    # Crear el contenido Markdown con tablas formateadas manualmente para permitir el uso de símbolos $ en los valores
    informe = f"""# Resultados del Modelo de Optimización: Pregunta 2

## Resumen de resultados

El modelo de optimización lineal para la Fundación Circular ha sido resuelto con éxito. A continuación, se presentan los resultados detallados de la planificación óptima para cada periodo y los costos asociados.

## 1. Planificación de producción y procesamiento

| Periodo | Ropa buen estado usada (kg) | Ropa mal estado procesada (kg) | Género utilizado (kg) | Prendas producidas | Demanda satisfecha | Demanda insatisfecha |
| :-----: | :------------------------: | :---------------------------: | :-------------------: | :----------------: | :----------------: | :------------------: |
|    1    |           ${df_prod.loc[0, 'Ropa buen estado usada (kg)']:.2f}$           |             ${df_prod.loc[0, 'Ropa mal estado procesada (kg)']:.2f}$             |         ${df_prod.loc[0, 'Género utilizado (kg)']:.2f}$          |       ${df_prod.loc[0, 'Prendas producidas']:.2f}$       |       ${df_prod.loc[0, 'Demanda satisfecha']:.2f}$       |        ${df_prod.loc[0, 'Demanda insatisfecha']:.2f}$        |
|    2    |           ${df_prod.loc[1, 'Ropa buen estado usada (kg)']:.2f}$           |             ${df_prod.loc[1, 'Ropa mal estado procesada (kg)']:.2f}$             |         ${df_prod.loc[1, 'Género utilizado (kg)']:.2f}$          |       ${df_prod.loc[1, 'Prendas producidas']:.2f}$       |       ${df_prod.loc[1, 'Demanda satisfecha']:.2f}$       |        ${df_prod.loc[1, 'Demanda insatisfecha']:.2f}$        |
|    3    |           ${df_prod.loc[2, 'Ropa buen estado usada (kg)']:.2f}$           |             ${df_prod.loc[2, 'Ropa mal estado procesada (kg)']:.2f}$             |         ${df_prod.loc[2, 'Género utilizado (kg)']:.2f}$          |       ${df_prod.loc[2, 'Prendas producidas']:.2f}$       |       ${df_prod.loc[2, 'Demanda satisfecha']:.2f}$       |        ${df_prod.loc[2, 'Demanda insatisfecha']:.2f}$        |
|    4    |           ${df_prod.loc[3, 'Ropa buen estado usada (kg)']:.2f}$           |             ${df_prod.loc[3, 'Ropa mal estado procesada (kg)']:.2f}$             |         ${df_prod.loc[3, 'Género utilizado (kg)']:.2f}$          |       ${df_prod.loc[3, 'Prendas producidas']:.2f}$       |       ${df_prod.loc[3, 'Demanda satisfecha']:.2f}$       |        ${df_prod.loc[3, 'Demanda insatisfecha']:.2f}$        |
|    5    |           ${df_prod.loc[4, 'Ropa buen estado usada (kg)']:.2f}$           |             ${df_prod.loc[4, 'Ropa mal estado procesada (kg)']:.2f}$             |         ${df_prod.loc[4, 'Género utilizado (kg)']:.2f}$          |       ${df_prod.loc[4, 'Prendas producidas']:.2f}$       |       ${df_prod.loc[4, 'Demanda satisfecha']:.2f}$       |        ${df_prod.loc[4, 'Demanda insatisfecha']:.2f}$        |
| **Total** | **${df_prod.loc[5, 'Ropa buen estado usada (kg)']:.2f}$** | **${df_prod.loc[5, 'Ropa mal estado procesada (kg)']:.2f}$** | **${df_prod.loc[5, 'Género utilizado (kg)']:.2f}$** | **${df_prod.loc[5, 'Prendas producidas']:.2f}$** | **${df_prod.loc[5, 'Demanda satisfecha']:.2f}$** | **${df_prod.loc[5, 'Demanda insatisfecha']:.2f}$** |

## 2. Inventarios al final de cada periodo

| Periodo | Inv. ropa buen estado (kg) | Inv. ropa mal estado (kg) | Inv. género (kg) | Almacenamiento total (kg) | % Capacidad utilizada |
| :-----: | :------------------------: | :-----------------------: | :--------------: | :-----------------------: | :-------------------: |
|    1    |            ${df_inv.loc[0, 'Inv. ropa buen estado (kg)']:.2f}$            |           ${df_inv.loc[0, 'Inv. ropa mal estado (kg)']:.2f}$            |       ${df_inv.loc[0, 'Inv. género (kg)']:.2f}$       |           ${df_inv.loc[0, 'Almacenamiento total (kg)']:.2f}$            |         ${df_inv.loc[0, '% Capacidad utilizada']:.2f}$          |
|    2    |            ${df_inv.loc[1, 'Inv. ropa buen estado (kg)']:.2f}$            |           ${df_inv.loc[1, 'Inv. ropa mal estado (kg)']:.2f}$            |       ${df_inv.loc[1, 'Inv. género (kg)']:.2f}$       |           ${df_inv.loc[1, 'Almacenamiento total (kg)']:.2f}$            |         ${df_inv.loc[1, '% Capacidad utilizada']:.2f}$          |
|    3    |            ${df_inv.loc[2, 'Inv. ropa buen estado (kg)']:.2f}$            |           ${df_inv.loc[2, 'Inv. ropa mal estado (kg)']:.2f}$            |       ${df_inv.loc[2, 'Inv. género (kg)']:.2f}$       |           ${df_inv.loc[2, 'Almacenamiento total (kg)']:.2f}$            |         ${df_inv.loc[2, '% Capacidad utilizada']:.2f}$          |
|    4    |            ${df_inv.loc[3, 'Inv. ropa buen estado (kg)']:.2f}$            |           ${df_inv.loc[3, 'Inv. ropa mal estado (kg)']:.2f}$            |       ${df_inv.loc[3, 'Inv. género (kg)']:.2f}$       |           ${df_inv.loc[3, 'Almacenamiento total (kg)']:.2f}$            |         ${df_inv.loc[3, '% Capacidad utilizada']:.2f}$          |
|    5    |            ${df_inv.loc[4, 'Inv. ropa buen estado (kg)']:.2f}$            |           ${df_inv.loc[4, 'Inv. ropa mal estado (kg)']:.2f}$            |       ${df_inv.loc[4, 'Inv. género (kg)']:.2f}$       |           ${df_inv.loc[4, 'Almacenamiento total (kg)']:.2f}$            |         ${df_inv.loc[4, '% Capacidad utilizada']:.2f}$          |

## 3. Recursos humanos y utilización

| Periodo | Trabajadores contratados | Trabajadores por boleta | Total trabajadores | Horas disponibles | Horas utilizadas | % Utilización |
| :-----: | :----------------------: | :---------------------: | :----------------: | :---------------: | :--------------: | :-----------: |
|    1    |            ${df_rec.loc[0, 'Trabajadores contratados']}$             |            ${df_rec.loc[0, 'Trabajadores por boleta']}$            |         ${df_rec.loc[0, 'Total trabajadores']}$          |       ${df_rec.loc[0, 'Horas disponibles']:.2f}$       |      ${df_rec.loc[0, 'Horas utilizadas']:.2f}$       |    ${df_rec.loc[0, '% Utilización']:.2f}$     |
|    2    |            ${df_rec.loc[1, 'Trabajadores contratados']}$             |            ${df_rec.loc[1, 'Trabajadores por boleta']}$            |         ${df_rec.loc[1, 'Total trabajadores']}$          |       ${df_rec.loc[1, 'Horas disponibles']:.2f}$       |      ${df_rec.loc[1, 'Horas utilizadas']:.2f}$       |    ${df_rec.loc[1, '% Utilización']:.2f}$     |
|    3    |            ${df_rec.loc[2, 'Trabajadores contratados']}$             |            ${df_rec.loc[2, 'Trabajadores por boleta']}$            |         ${df_rec.loc[2, 'Total trabajadores']}$          |       ${df_rec.loc[2, 'Horas disponibles']:.2f}$       |      ${df_rec.loc[2, 'Horas utilizadas']:.2f}$       |    ${df_rec.loc[2, '% Utilización']:.2f}$     |
|    4    |            ${df_rec.loc[3, 'Trabajadores contratados']}$             |            ${df_rec.loc[3, 'Trabajadores por boleta']}$            |         ${df_rec.loc[3, 'Total trabajadores']}$          |       ${df_rec.loc[3, 'Horas disponibles']:.2f}$       |      ${df_rec.loc[3, 'Horas utilizadas']:.2f}$       |    ${df_rec.loc[3, '% Utilización']:.2f}$     |
|    5    |            ${df_rec.loc[4, 'Trabajadores contratados']}$             |            ${df_rec.loc[4, 'Trabajadores por boleta']}$            |         ${df_rec.loc[4, 'Total trabajadores']}$          |       ${df_rec.loc[4, 'Horas disponibles']:.2f}$       |      ${df_rec.loc[4, 'Horas utilizadas']:.2f}$       |    ${df_rec.loc[4, '% Utilización']:.2f}$     |
| **Total** | **${df_rec.loc[5, 'Trabajadores contratados']}$** | **${df_rec.loc[5, 'Trabajadores por boleta']}$** | **${df_rec.loc[5, 'Total trabajadores']}$** | **${df_rec.loc[5, 'Horas disponibles']:.2f}$** | **${df_rec.loc[5, 'Horas utilizadas']:.2f}$** | **${df_rec.loc[5, '% Utilización']:.2f}$** |

## 4. Desglose de costos

| Componente | Fórmula | Valor ($) | Porcentaje |
| :--------: | :-----: | :-------: | :--------: |
| {df_costos.loc[0, 'Componente']} | ${df_costos.loc[0, 'Fórmula']}$ | ${df_costos.loc[0, 'Valor ($)']:,.2f}$ | ${df_costos.loc[0, 'Porcentaje']:.2f}\\%$ |
| {df_costos.loc[1, 'Componente']} | ${df_costos.loc[1, 'Fórmula']}$ | ${df_costos.loc[1, 'Valor ($)']:,.2f}$ | ${df_costos.loc[1, 'Porcentaje']:.2f}\\%$ |
| {df_costos.loc[2, 'Componente']} | ${df_costos.loc[2, 'Fórmula']}$ | ${df_costos.loc[2, 'Valor ($)']:,.2f}$ | ${df_costos.loc[2, 'Porcentaje']:.2f}\\%$ |
| {df_costos.loc[3, 'Componente']} | ${df_costos.loc[3, 'Fórmula']}$ | ${df_costos.loc[3, 'Valor ($)']:,.2f}$ | ${df_costos.loc[3, 'Porcentaje']:.2f}\\%$ |
| {df_costos.loc[4, 'Componente']} | ${df_costos.loc[4, 'Fórmula']}$ | ${df_costos.loc[4, 'Valor ($)']:,.2f}$ | ${df_costos.loc[4, 'Porcentaje']:.2f}\\%$ |
| {df_costos.loc[5, 'Componente']} | ${df_costos.loc[5, 'Fórmula']}$ | ${df_costos.loc[5, 'Valor ($)']:,.2f}$ | ${df_costos.loc[5, 'Porcentaje']:.2f}\\%$ |
| {df_costos.loc[6, 'Componente']} | ${df_costos.loc[6, 'Fórmula']}$ | ${df_costos.loc[6, 'Valor ($)']:,.2f}$ | ${df_costos.loc[6, 'Porcentaje']:.2f}\\%$ |

## 5. Visualizaciones

### 5.1 Producción vs Demanda

![Producción vs Demanda](../src/pregunta2/grafico_produccion_demanda_p2.png)

*Figura 1: Producción vs. demanda por periodo, incluyendo análisis de demanda satisfecha e insatisfecha con indicadores de porcentaje.*

### 5.2 Uso de Capacidad de Almacenamiento

![Uso de Capacidad](../src/pregunta2/grafico_uso_capacidad_p2.png)

*Figura 2: Uso de capacidad de almacenamiento por periodo, mostrando distribución por tipo de inventario y porcentaje de utilización de la capacidad total.*

### 5.3 Distribución de Costos

![Distribución de Costos](../src/pregunta2/grafico_distribucion_costos_p2.png)

*Figura 3: Distribución porcentual y absoluta de los componentes de costo, destacando los principales factores que contribuyen al costo total.*

### 5.4 Recursos Humanos y Utilización

![Recursos Humanos](../src/pregunta2/grafico_recursos_humanos_p2.png)

*Figura 4: Distribución de recursos humanos y porcentaje de utilización por periodo, desglosando entre trabajadores contratados y por boleta.*

## 6. Análisis de la solución óptima

### 6.1 Estrategia óptima de producción

Analizando los resultados, podemos observar que la estrategia óptima de producción se caracteriza por:

- **Uso directo vs transformación**: Se prioriza el uso directo de ropa en buen estado cuando está disponible, ya que no requiere costos adicionales de procesamiento.
- **Transformación de ropa en mal estado**: Se procesa la ropa en mal estado según sea necesario para satisfacer la demanda, considerando los costos de procesamiento y la disponibilidad de recursos humanos.
- **Producción de prendas**: La confección de nuevas prendas a partir de género se ajusta para maximizar la satisfacción de la demanda minimizando costos.

### 6.2 Gestión de inventarios

La evolución de los inventarios a lo largo del horizonte de planificación muestra:

- **Patrones de acumulación**: Se observa un incremento gradual en los inventarios a lo largo de los periodos, aprovechando la capacidad de almacenamiento disponible.
- **Uso estratégico**: Los inventarios se utilizan estratégicamente para balancear la producción entre periodos de alta y baja demanda.
- **Restricción de capacidad**: El almacenamiento total se mantiene siempre por debajo de la capacidad máxima, con un uso promedio del ${df_inv['% Capacidad utilizada'].mean():.2f}$% de la capacidad disponible.

### 6.3 Recursos humanos

El patrón de contratación de trabajadores por boleta revela:

- **Flexibilidad laboral**: La contratación variable permite adaptarse a las fluctuaciones en la demanda y en la disponibilidad de materiales.
- **Eficiencia en el uso**: Se logra un porcentaje de utilización promedio del ${utilizacion_promedio:.2f}$% de las horas-hombre disponibles.

### 6.4 Componentes principales del costo

El análisis de costos muestra que:

- **Mayor componente**: El componente "{componente_mayor_costo['Componente']}" representa el mayor porcentaje del costo total con un ${componente_mayor_costo['Porcentaje']:.2f}$%.
- **Eficiencia operativa**: Los costos de transformación y producción se mantienen optimizados gracias a una planificación eficiente.
- **Penalizaciones**: La demanda no satisfecha genera un costo de penalización que representa el ${df_costos.loc[5, 'Porcentaje']:.2f}$% del costo total.
- **Costos laborales**: Los costos relacionados con personal (contratado y por boleta) representan conjuntamente el ${df_costos.loc[0, 'Porcentaje'] + df_costos.loc[1, 'Porcentaje']:.2f}$% del costo total.

### 6.5 Análisis adicional con visualizaciones detalladas

Las visualizaciones proporcionan información adicional que ayuda a interpretar los resultados del modelo:

- **Capacidad de satisfacción de demanda**: Se puede observar que el porcentaje de demanda satisfecha fluctúa entre periodos, con un promedio cercano al ${porcentaje_demanda_sat:.2f}$%. El periodo ${df_prod[df_prod['Periodo'] != 'Total']['Demanda satisfecha'].idxmax() + 1}$ muestra la mayor satisfacción de demanda en términos absolutos.

- **Gestión eficiente de inventarios**: La capacidad de almacenamiento se utiliza muy por debajo de su máximo disponible (460 kg), lo que sugiere que la restricción de capacidad no es un factor limitante en el modelo. El periodo con mayor nivel de inventario presenta apenas un ${df_inv['% Capacidad utilizada'].max():.2f}$% de la capacidad total utilizada.

- **Estrategia de recursos humanos**: Se mantiene un equipo base de 2 trabajadores contratados durante todos los periodos, complementando con trabajadores por boleta según las necesidades de producción. Esta estrategia optimiza los costos laborales manteniendo una alta utilización (${utilizacion_promedio:.2f}$%) del tiempo disponible.

- **Oportunidades de mejora**: El alto porcentaje de demanda insatisfecha (${100-porcentaje_demanda_sat:.2f}$%) y su consecuente costo de penalización sugieren que podría ser beneficioso evaluar alternativas como:
  * Aumentar la capacidad productiva mediante más trabajadores
  * Mejorar la eficiencia de los procesos de transformación
  * Revisar la estrategia de adquisición de materiales

## 7. Conclusiones

El modelo de optimización ha proporcionado una planificación detallada y eficiente para la operación de la Fundación Circular, permitiendo:

1. **Maximizar el aprovechamiento de recursos** donados de ropa en buen y mal estado.
2. **Minimizar los costos operativos** manteniendo un balance adecuado entre producción directa y transformación.
3. **Gestionar eficientemente el personal** mediante la contratación estratégica de trabajadores por boleta.
4. **Optimizar el uso del almacenamiento disponible** sin exceder la capacidad máxima.

Esta planificación óptima permite a la Fundación Circular cumplir con su objetivo social de manera económicamente sostenible.
"""
    
    # Guardar el informe en un archivo Markdown
    os.makedirs(RESPUESTAS_DIR, exist_ok=True)
    with open(RESPUESTAS_DIR / 'pregunta2_resolucion.md', 'w') as f:
        f.write(informe)
    
    return informe

if __name__ == "__main__":
    print("Resolviendo el modelo de optimización para la pregunta 2...")
    
    # Resolver el modelo
    resultados, param, T, kb, km, d = resolver_modelo_base()
    
    if resultados:
        print(f"El modelo se ha resuelto correctamente. Valor óptimo: ${resultados['valor_objetivo']:.2f}")
        
        # Generar tablas de resultados
        tablas = generar_informe(resultados, param, T, kb, km, d)
        
        # Generar visualizaciones mejoradas
        print("Generando gráficos mejorados...")
        generar_graficos(tablas)
        
        # Generar informe en Markdown
        print("Generando informe markdown...")
        informe_markdown = generar_informe_markdown(tablas)
        
        print("Resultados generados con éxito.")
        print(f"Valor óptimo de la función objetivo: ${resultados['valor_objetivo']:.2f}")
        
        # Mostrar un resumen de los resultados
        print("\nResumen de resultados:")
        
        # Producción total
        prendas_producidas = tablas['produccion'].loc[tablas['produccion']['Periodo'] == 'Total', 'Prendas producidas'].values[0]
        demanda_satisfecha = tablas['produccion'].loc[tablas['produccion']['Periodo'] == 'Total', 'Demanda satisfecha'].values[0]
        demanda_insatisfecha = tablas['produccion'].loc[tablas['produccion']['Periodo'] == 'Total', 'Demanda insatisfecha'].values[0]
        
        print(f"- Total de prendas producidas: {prendas_producidas:.2f}")
        print(f"- Demanda satisfecha: {demanda_satisfecha:.2f} ({demanda_satisfecha/(demanda_satisfecha+demanda_insatisfecha)*100:.2f}%)")
        print(f"- Demanda insatisfecha: {demanda_insatisfecha:.2f} ({demanda_insatisfecha/(demanda_satisfecha+demanda_insatisfecha)*100:.2f}%)")
        
        # Componente de mayor costo
        mayor_componente = tablas['costos'].iloc[:-1].loc[tablas['costos'].iloc[:-1]['Valor ($)'].idxmax()]
        print(f"- Mayor componente de costo: {mayor_componente['Componente']} (${mayor_componente['Valor ($)']:.2f}, {mayor_componente['Porcentaje']:.2f}%)")
        
        print(f"\nSe ha generado el archivo: {RESPUESTAS_DIR}/pregunta2_resolucion.md")
        
    else:
        print("No se pudo resolver el modelo. Verifique los datos y restricciones.")
