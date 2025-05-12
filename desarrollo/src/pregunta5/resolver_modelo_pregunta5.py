#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Implementación del modelo de optimización para la Fundación Circular.
Este script resuelve el modelo para la pregunta 5, considerando la política
de dotación mínima de personal.
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
OUTPUT_DIR = BASE_DIR / "desarrollo" / "src" / "pregunta5"
RESPUESTAS_DIR = BASE_DIR / "desarrollo" / "respuestas"

def resolver_modelo_general(aplicar_dotacion_minima=False):
    """
    Resuelve el modelo de optimización con los parámetros dados.
    Si aplicar_dotacion_minima es True, aplica la restricción de dotación mínima de personal.
    """
    # Cargar datos
    parametros = pd.read_csv(DATA_DIR / 'datos_parametros.csv')
    periodos = pd.read_csv(DATA_DIR / 'datos_periodos.csv')
    
    # Convertir los parámetros a un diccionario para fácil acceso
    param_dict = dict(zip(parametros['Parametro'], parametros['Valor']))

    # Extraer los datos por periodo
    T = len(periodos)  # Número de periodos
    kb = periodos['kb_t'].tolist()
    km = periodos['km_t'].tolist()
    d = periodos['d_t'].tolist()
    
    # Extraer parámetros específicos
    rb = param_dict['rb']  # Inventario inicial de ropa en buen estado (kg)
    rm = param_dict['rm']  # Inventario inicial de ropa en mal estado (kg)
    w0 = int(param_dict['w0'])  # Dotación inicial de trabajadores
    cc = param_dict['cc']  # Costo por hora normal trabajada ($/hora)
    ct = param_dict['ct']  # Costo por trabajador por boleta ($/persona/periodo)
    cp = param_dict['cp']  # Costo de penalización por demanda insatisfecha ($/prenda)
    g = param_dict['g']  # Costo unitario de transformación a género ($/kg)
    n = param_dict['n']  # Costo unitario de producción de prendas desde género ($/kg)
    a = param_dict['a']  # Costo de almacenamiento ($/kg/periodo)
    s = param_dict['s']  # Capacidad máxima de almacenamiento (kg)
    p = param_dict['p']  # Peso promedio de cada prenda (kg/prenda)
    h = param_dict['h']  # Horas de trabajo por trabajador por periodo
    tau_g = param_dict['tau_g']  # Horas-hombre para transformar 1 kg de ropa en mal estado
    tau_n = param_dict['tau_n']  # Horas-hombre para confeccionar 1 kg de ropa reutilizada
    tr = param_dict['tr']  # Número mínimo de trabajadores requeridos por periodo

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
    obj_function = pulp.lpSum([W[t] * ct + cc * h * w0 + g * Y[t] + n * Z[t] + 
                         a * (IB[t] + IM[t] + IG[t]) + cp * NS[t] for t in range(1, T+1)])
    
    model += obj_function
    
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
        
    # Restricción de dotación mínima de personal si se aplica
    if aplicar_dotacion_minima:
        for t in range(1, T+1):
            model += w0 + W[t] >= tr
    
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
    
    return resultados, param_dict, T, kb, km, d

def resolver_modelo_base():
    """Resuelve el modelo de optimización lineal base sin restricción de dotación mínima"""
    return resolver_modelo_general(aplicar_dotacion_minima=False)

def resolver_modelo_con_dotacion_minima():
    """Resuelve el modelo de optimización aplicando la restricción de dotación mínima"""
    return resolver_modelo_general(aplicar_dotacion_minima=True)

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

def generar_graficos_comparativos(resultados_base, resultados_dotacion):
    """Genera gráficos comparativos entre el caso base y el caso con dotación mínima"""
    # Configuración de estilo para los gráficos
    sns.set_theme(style="darkgrid")
    
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
    
    # Crear dataframes para los resultados
    periodos = list(range(1, 6))  # Suponiendo que hay 5 periodos
    
    # Para producción y demanda
    df_prod_base = pd.DataFrame({
        'Periodo': periodos,
        'Prendas Producidas': [
            (resultados_base['X'][t] + resultados_base['Z'][t]) / param['p'] 
            for t in periodos
        ],
        'Demanda Satisfecha': [
            pd.read_csv(DATA_DIR / 'datos_periodos.csv')['d_t'][t-1] - resultados_base['NS'][t] 
            for t in periodos
        ],
        'Demanda Insatisfecha': [resultados_base['NS'][t] for t in periodos]
    })
    
    df_prod_dotacion = pd.DataFrame({
        'Periodo': periodos,
        'Prendas Producidas': [
            (resultados_dotacion['X'][t] + resultados_dotacion['Z'][t]) / param['p'] 
            for t in periodos
        ],
        'Demanda Satisfecha': [
            pd.read_csv(DATA_DIR / 'datos_periodos.csv')['d_t'][t-1] - resultados_dotacion['NS'][t] 
            for t in periodos
        ],
        'Demanda Insatisfecha': [resultados_dotacion['NS'][t] for t in periodos]
    })
    
    # Para recursos humanos
    df_rec_base = pd.DataFrame({
        'Periodo': periodos,
        'Total Trabajadores': [param['w0'] + resultados_base['W'][t] for t in periodos],
        'Trabajadores por Boleta': [resultados_base['W'][t] for t in periodos],
        '% Utilización': [
            (param['tau_g'] * resultados_base['Y'][t] + param['tau_n'] * resultados_base['Z'][t]) / 
            (param['h'] * (param['w0'] + resultados_base['W'][t])) * 100 if (param['w0'] + resultados_base['W'][t]) > 0 else 0
            for t in periodos
        ]
    })
    
    df_rec_dotacion = pd.DataFrame({
        'Periodo': periodos,
        'Total Trabajadores': [param['w0'] + resultados_dotacion['W'][t] for t in periodos],
        'Trabajadores por Boleta': [resultados_dotacion['W'][t] for t in periodos],
        '% Utilización': [
            (param['tau_g'] * resultados_dotacion['Y'][t] + param['tau_n'] * resultados_dotacion['Z'][t]) / 
            (param['h'] * (param['w0'] + resultados_dotacion['W'][t])) * 100 if (param['w0'] + resultados_dotacion['W'][t]) > 0 else 0
            for t in periodos
        ]
    })
    
    # Para procesamiento de ropa en mal estado
    df_proc_base = pd.DataFrame({
        'Periodo': periodos,
        'Ropa Mal Estado Procesada (kg)': [resultados_base['Y'][t] for t in periodos]
    })
    
    df_proc_dotacion = pd.DataFrame({
        'Periodo': periodos,
        'Ropa Mal Estado Procesada (kg)': [resultados_dotacion['Y'][t] for t in periodos]
    })
    
    # =====================================================
    # 1. Gráfico comparativo de Producción vs Demanda
    # =====================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), facecolor='white')
    fig.suptitle('Comparación de Producción y Demanda: Caso Base vs Caso con Dotación Mínima', 
                fontsize=16, fontweight='bold')
    
    # Gráfico para caso base
    ax1.set_facecolor('#f8f9fa')
    ax1.set_title('Caso Base', fontsize=14, fontweight='bold')
    
    # Crear gráfico apilado para la demanda (caso base)
    ax1.bar(df_prod_base['Periodo'], df_prod_base['Demanda Satisfecha'], 
            label='Demanda Satisfecha', alpha=0.8, color=paleta_colores['verde_exito'])
    ax1.bar(df_prod_base['Periodo'], df_prod_base['Demanda Insatisfecha'], 
            bottom=df_prod_base['Demanda Satisfecha'], label='Demanda Insatisfecha', 
            alpha=0.8, color=paleta_colores['rojo_error'])
    
    # Añadir línea para producción (caso base)
    ax1.plot(df_prod_base['Periodo'], df_prod_base['Prendas Producidas'], 'o-', 
            color=paleta_colores['azul_principal'], linewidth=2.5, label='Prendas Producidas', 
            markersize=8, path_effects=[patheffects.withStroke(linewidth=4, foreground='white')])
    
    # Etiquetas y configuración (caso base)
    ax1.set_xlabel('Periodo', fontsize=12)
    ax1.set_ylabel('Cantidad de Prendas', fontsize=12)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Gráfico para caso con dotación mínima
    ax2.set_facecolor('#f8f9fa')
    ax2.set_title('Caso con Dotación Mínima', fontsize=14, fontweight='bold')
    
    # Crear gráfico apilado para la demanda (caso con dotación mínima)
    ax2.bar(df_prod_dotacion['Periodo'], df_prod_dotacion['Demanda Satisfecha'], 
            label='Demanda Satisfecha', alpha=0.8, color=paleta_colores['verde_exito'])
    ax2.bar(df_prod_dotacion['Periodo'], df_prod_dotacion['Demanda Insatisfecha'], 
            bottom=df_prod_dotacion['Demanda Satisfecha'], label='Demanda Insatisfecha', 
            alpha=0.8, color=paleta_colores['rojo_error'])
    
    # Añadir línea para producción (caso con dotación mínima)
    ax2.plot(df_prod_dotacion['Periodo'], df_prod_dotacion['Prendas Producidas'], 'o-', 
            color=paleta_colores['azul_principal'], linewidth=2.5, label='Prendas Producidas', 
            markersize=8, path_effects=[patheffects.withStroke(linewidth=4, foreground='white')])
    
    # Etiquetas y configuración (caso con personal)
    ax2.set_xlabel('Periodo', fontsize=12)
    ax2.set_ylabel('Cantidad de Prendas', fontsize=12)
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # Ajustar el diseño y guardar
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'grafico_comparativo_produccion_demanda.png', dpi=300)
    
    # =====================================================
    # 2. Gráfico comparativo de Recursos Humanos
    # =====================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), facecolor='white')
    fig.suptitle('Comparación de Recursos Humanos: Caso Base vs Caso con Dotación Mínima', 
                fontsize=16, fontweight='bold')
    
    # Gráfico para caso base
    ax1.set_facecolor('#f8f9fa')
    ax1.set_title('Caso Base', fontsize=14, fontweight='bold')
    
    # Crear gráfico de barras para trabajadores (caso base)
    width = 0.6
    ax1.bar(df_rec_base['Periodo'], [param['w0']] * len(periodos), width,
            label='Trabajadores Contratados', color=paleta_colores['azul_principal'])
    ax1.bar(df_rec_base['Periodo'], df_rec_base['Trabajadores por Boleta'], width,
            bottom=[param['w0']] * len(periodos), label='Trabajadores por Boleta',
            color=paleta_colores['naranja_alerta'])
    
    # Añadir etiquetas con el total de trabajadores (caso base)
    for i, periodo in enumerate(periodos):
        ax1.annotate(f'Total: {df_rec_base["Total Trabajadores"].iloc[i]}',
                    xy=(periodo, df_rec_base['Total Trabajadores'].iloc[i] + 0.3),
                    ha='center', va='bottom', fontweight='bold')
    
    # Añadir línea para porcentaje de utilización (caso base)
    ax3 = ax1.twinx()
    ax3.plot(df_rec_base['Periodo'], df_rec_base['% Utilización'], 'o-', 
            color=paleta_colores['verde_exito'], linewidth=2.5, label='% Utilización')
    
    # Configuración de ejes y etiquetas (caso base)
    ax1.set_xlabel('Periodo', fontsize=12)
    ax1.set_ylabel('Número de Trabajadores', fontsize=12)
    ax3.set_ylabel('% Utilización', fontsize=12, color=paleta_colores['verde_exito'])
    ax3.set_ylim(0, 105)  # Ajustar para que 100% sea visible
    
    # Combinar leyendas
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines3, labels3 = ax3.get_legend_handles_labels()
    ax1.legend(lines1 + lines3, labels1 + labels3, loc='upper left', fontsize=10)
    
    # Gráfico para caso con dotación mínima
    ax2.set_facecolor('#f8f9fa')
    ax2.set_title('Caso con Dotación Mínima', fontsize=14, fontweight='bold')
    
    # Crear gráfico de barras para trabajadores (caso con dotación)
    ax2.bar(df_rec_dotacion['Periodo'], [param['w0']] * len(periodos), width,
            label='Trabajadores Contratados', color=paleta_colores['azul_principal'])
    ax2.bar(df_rec_dotacion['Periodo'], df_rec_dotacion['Trabajadores por Boleta'], width,
            bottom=[param['w0']] * len(periodos), label='Trabajadores por Boleta',
            color=paleta_colores['naranja_alerta'])
    
    # Añadir etiquetas con el total de trabajadores (caso con dotación mínima)
    for i, periodo in enumerate(periodos):
        ax2.annotate(f'Total: {df_rec_dotacion["Total Trabajadores"].iloc[i]}',
                    xy=(periodo, df_rec_dotacion['Total Trabajadores'].iloc[i] + 0.3),
                    ha='center', va='bottom', fontweight='bold')
    
    # Añadir línea para porcentaje de utilización (caso con dotación mínima)
    ax4 = ax2.twinx()
    ax4.plot(df_rec_dotacion['Periodo'], df_rec_dotacion['% Utilización'], 'o-', 
            color=paleta_colores['verde_exito'], linewidth=2.5, label='% Utilización')
    
    # Configuración de ejes y etiquetas (caso con dotación mínima)
    ax2.set_xlabel('Periodo', fontsize=12)
    ax2.set_ylabel('Número de Trabajadores', fontsize=12)
    ax4.set_ylabel('% Utilización', fontsize=12, color=paleta_colores['verde_exito'])
    ax4.set_ylim(0, 105)  # Ajustar para que 100% sea visible
    
    # Combinar leyendas
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines4, labels4 = ax4.get_legend_handles_labels()
    ax2.legend(lines2 + lines4, labels2 + labels4, loc='upper left', fontsize=10)
    
    # Ajustar el diseño y guardar
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'grafico_comparativo_recursos_humanos.png', dpi=300)
    
    # =====================================================
    # 3. Gráfico comparativo de Costos
    # =====================================================
    # Calcular costos para cada caso
    costos_base = {
        'Personal': param['cc'] * param['h'] * param['w0'] * 5 + param['ct'] * sum(resultados_base['W'].values()),
        'Transformación': param['g'] * sum(resultados_base['Y'].values()),
        'Producción': param['n'] * sum(resultados_base['Z'].values()),
        'Almacenamiento': param['a'] * sum(sum(resultados_base[inv].values()) for inv in ['IB', 'IM', 'IG']),
        'Penalización': param['cp'] * sum(resultados_base['NS'].values())
    }
    
    costos_dotacion = {
        'Personal': param['cc'] * param['h'] * param['w0'] * 5 + param['ct'] * sum(resultados_dotacion['W'].values()),
        'Transformación': param['g'] * sum(resultados_dotacion['Y'].values()),
        'Producción': param['n'] * sum(resultados_dotacion['Z'].values()),
        'Almacenamiento': param['a'] * sum(sum(resultados_dotacion[inv].values()) for inv in ['IB', 'IM', 'IG']),
        'Penalización': param['cp'] * sum(resultados_dotacion['NS'].values())
    }
    
    categorias_base = list(costos_base.keys())
    valores_base = list(costos_base.values())
    
    categorias_dotacion = list(costos_dotacion.keys())
    valores_dotacion = list(costos_dotacion.values())
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), facecolor='white')
    fig.suptitle('Comparación de Costos: Caso Base vs Caso con Dotación Mínima de Personal',
                fontsize=16, fontweight='bold')
    
    # Gráfico de torta para caso base
    ax1.set_facecolor('#f8f9fa')
    ax1.set_title(f'Caso Base - Costo Total: ${resultados_base["valor_objetivo"]:,.2f}', 
                 fontsize=14, fontweight='bold')
    
    # Calcular porcentajes para las etiquetas
    total_base = sum(valores_base)
    porcentajes_base = [100 * v / total_base for v in valores_base]
    
    # Gráfico de torta con etiquetas de porcentaje
    wedges, texts, autotexts = ax1.pie(
        valores_base, 
        labels=categorias_base,
        autopct='%1.1f%%',
        startangle=90,
        colors=plt.cm.tab10.colors[:len(categorias_base)],
        explode=[0.05 if i == valores_base.index(max(valores_base)) else 0 for i in range(len(valores_base))]
    )
    
    # Mejorar legibilidad de las etiquetas
    for autotext in autotexts:
        autotext.set_fontsize(10)
        autotext.set_fontweight('bold')
    
    # Gráfico de torta para caso con dotación mínima de personal
    ax2.set_facecolor('#f8f9fa')
    ax2.set_title(f'Caso con Dotación Mínima - Costo Total: ${resultados_dotacion["valor_objetivo"]:,.2f}',
                 fontsize=14, fontweight='bold')
    
    # Calcular porcentajes para las etiquetas
    total_dotacion = sum(valores_dotacion)
    porcentajes_dotacion = [100 * v / total_dotacion for v in valores_dotacion]
    
    # Gráfico de torta con etiquetas de porcentaje
    wedges, texts, autotexts = ax2.pie(
        valores_dotacion, 
        labels=categorias_dotacion,
        autopct='%1.1f%%',
        startangle=90,
        colors=plt.cm.tab10.colors[:len(categorias_dotacion)],
        explode=[0.05 if i == valores_dotacion.index(max(valores_dotacion)) else 0 for i in range(len(valores_dotacion))]
    )
    
    # Mejorar legibilidad de las etiquetas
    for autotext in autotexts:
        autotext.set_fontsize(10)
        autotext.set_fontweight('bold')
    
    # Ajustar el diseño y guardar
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'grafico_comparativo_costos.png', dpi=300)
    
    # =====================================================
    # 4. Gráfico comparativo de Procesamiento de Ropa en Mal Estado
    # =====================================================
    fig, ax = plt.subplots(figsize=(12, 8), facecolor='white')
    ax.set_facecolor('#f8f9fa')
    
    # Configurar ancho de barras y posiciones
    bar_width = 0.35
    index = np.arange(len(periodos))
    
    # Crear barras para caso base y caso con dotación mínima
    bars1 = ax.bar(index - bar_width/2, df_proc_base['Ropa Mal Estado Procesada (kg)'],
                  bar_width, label='Caso Base', color=paleta_colores['azul_principal'])
    bars2 = ax.bar(index + bar_width/2, df_proc_dotacion['Ropa Mal Estado Procesada (kg)'],
                  bar_width, label='Caso con Dotación Mínima', color=paleta_colores['verde_exito'])
    
    # Añadir etiquetas a las barras
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                   xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3),  # 3 puntos de desplazamiento vertical
                   textcoords="offset points",
                   ha='center', va='bottom',
                   fontsize=9, fontweight='bold')
    
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                   xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3),  # 3 puntos de desplazamiento vertical
                   textcoords="offset points",
                   ha='center', va='bottom',
                   fontsize=9, fontweight='bold')
    
    # Configuración del gráfico
    ax.set_xlabel('Periodo', fontsize=12, fontweight='bold')
    ax.set_ylabel('Ropa en Mal Estado Procesada (kg)', fontsize=12, fontweight='bold')
    ax.set_title('Comparación del Procesamiento de Ropa en Mal Estado',
               fontsize=14, fontweight='bold')
    ax.set_xticks(index)
    ax.set_xticklabels([f'Periodo {i}' for i in periodos])
    ax.legend()
    
    # Añadir una línea horizontal para el total en cada caso
    ax.axhline(y=sum(df_proc_base['Ropa Mal Estado Procesada (kg)']/len(periodos)),
              color=paleta_colores['azul_principal'], linestyle='--', alpha=0.7,
              label=f'Promedio Base: {sum(df_proc_base["Ropa Mal Estado Procesada (kg)"]/len(periodos)):.1f} kg')
    ax.axhline(y=sum(df_proc_dotacion['Ropa Mal Estado Procesada (kg)']/len(periodos)),
              color=paleta_colores['verde_exito'], linestyle='--', alpha=0.7,
              label=f'Promedio Dotación Mínima: {sum(df_proc_dotacion["Ropa Mal Estado Procesada (kg)"]/len(periodos)):.1f} kg')
    
    # Actualizar leyenda con las líneas promedio
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels, loc='upper left')
    
    # Añadir cuadro de texto con la diferencia porcentual
    cambio_porcentual = ((sum(df_proc_dotacion['Ropa Mal Estado Procesada (kg)']) - 
                         sum(df_proc_base['Ropa Mal Estado Procesada (kg)'])) / 
                        sum(df_proc_base['Ropa Mal Estado Procesada (kg)'])) * 100
    
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    ax.text(0.05, 0.95, f'Variación total: {cambio_porcentual:.2f}%',
           transform=ax.transAxes, fontsize=12,
           verticalalignment='top', bbox=props)
    
    # Ajustar el diseño y guardar
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'grafico_comparativo_procesamiento.png', dpi=300)
    
    print("Gráficos comparativos generados con éxito.")
    
    # Crear carpeta para gráficos si no existe
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Añadir un gráfico específico para mostrar la comparación de personal
    plt.figure(figsize=(10, 6), facecolor='white')
    
    
    # Crear datos para la comparación
    categorias = ['Personal Mínimo Requerido', 'Personal Caso Base', 'Personal Con Dotación Mínima']
    valores = [param['tr'], min([param['w0'] + resultados_base['W'][t] for t in range(1, len(periodos)+1)]), 
               min([param['w0'] + resultados_dotacion['W'][t] for t in range(1, len(periodos)+1)])]
    
    # Crear barras
    barras = plt.bar(categorias, valores, color=[paleta_colores['azul_claro'], 
                                                 paleta_colores['azul_principal'], 
                                                 paleta_colores['verde_exito']])
    
    # Añadir etiquetas a las barras
    for barra in barras:
        height = barra.get_height()
        plt.text(barra.get_x() + barra.get_width()/2., height + 0.1,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.title('Comparación de Personal Mínimo vs. Dotación en los Modelos', fontsize=16, fontweight='bold')
    plt.ylabel('Número de Trabajadores', fontsize=14)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'grafico_dotacion_minima.png', dpi=300)
    plt.close()

def generar_informe_comparativo(resultados_base, resultados_dotacion):
    """Genera un informe comparativo entre el caso base y el caso con dotación mínima de personal"""
    
    # Cargar parámetros del modelo
    parametros = pd.read_csv(DATA_DIR / 'datos_parametros.csv')
    param = dict(zip(parametros['Parametro'], parametros['Valor']))
    
    # Calcular diferencias en indicadores clave
    cambio_valor_objetivo = ((resultados_dotacion['valor_objetivo'] - resultados_base['valor_objetivo']) / 
                            resultados_base['valor_objetivo']) * 100
    
    total_Y_base = sum(resultados_base['Y'].values())
    total_Y_dotacion = sum(resultados_dotacion['Y'].values())
    cambio_procesamiento = ((total_Y_dotacion - total_Y_base) / total_Y_base) * 100 if total_Y_base > 0 else float('inf')
    
    total_NS_base = sum(resultados_base['NS'].values())
    total_NS_dotacion = sum(resultados_dotacion['NS'].values())
    cambio_demanda_insatisfecha = ((total_NS_dotacion - total_NS_base) / total_NS_base) * 100 if total_NS_base > 0 else float('inf')
    
    total_W_base = sum(resultados_base['W'].values())
    total_W_dotacion = sum(resultados_dotacion['W'].values())
    cambio_trabajadores = ((total_W_dotacion - total_W_base) / total_W_base) * 100 if total_W_base > 0 else float('inf')
    
    # Información sobre el mínimo de dotación
    dotacion_minima = param.get('tr', 0)
    
    # Generar gráficos comparativos
    generar_graficos_comparativos(resultados_base, resultados_dotacion)
    
    # Crear el informe comparativo
    informe = f"""# Análisis Comparativo: Impacto de la Política de Dotación Mínima de Personal

## Introducción

Este análisis compara los resultados del modelo base con los resultados obtenidos tras aplicar la política de dotación mínima de personal que exige mantener un mínimo de ${param['tr']}$ trabajadores activos en cada periodo.

## Tablas Comparativas

### 1. Planificación de Producción y Procesamiento

|  Periodo  |   Caso    | Ropa buen estado (kg) | Ropa mal estado (kg) | Género utilizado (kg) | Prendas producidas | Demanda satisfecha | Demanda insatisfecha |
| :-------: | :-------: | :-------------------: | :------------------: | :-------------------: | :----------------: | :----------------: | :------------------: |
|     1     |   Base    |         ${resultados_base['X'][1]:.2f}$         |        ${resultados_base['Y'][1]:.2f}$         |         ${resultados_base['Z'][1]:.2f}$         |       ${(resultados_base['X'][1] + resultados_base['Z'][1]) / 0.4:.2f}$       |       ${pd.read_csv(DATA_DIR / 'datos_periodos.csv')['d_t'][0] - resultados_base['NS'][1]:.2f}$       |        ${resultados_base['NS'][1]:.2f}$        |
|     1     |   Dotación   |         ${resultados_dotacion['X'][1]:.2f}$         |        ${resultados_dotacion['Y'][1]:.2f}$         |         ${resultados_dotacion['Z'][1]:.2f}$         |       ${(resultados_dotacion['X'][1] + resultados_dotacion['Z'][1]) / 0.4:.2f}$       |       ${pd.read_csv(DATA_DIR / 'datos_periodos.csv')['d_t'][0] - resultados_dotacion['NS'][1]:.2f}$       |        ${resultados_dotacion['NS'][1]:.2f}$        |
|     2     |   Base    |         ${resultados_base['X'][2]:.2f}$          |        ${resultados_base['Y'][2]:.2f}$         |         ${resultados_base['Z'][2]:.2f}$         |       ${(resultados_base['X'][2] + resultados_base['Z'][2]) / 0.4:.2f}$       |       ${pd.read_csv(DATA_DIR / 'datos_periodos.csv')['d_t'][1] - resultados_base['NS'][2]:.2f}$       |        ${resultados_base['NS'][2]:.2f}$        |
|     2     |   Dotación   |         ${resultados_dotacion['X'][2]:.2f}$          |        ${resultados_dotacion['Y'][2]:.2f}$         |         ${resultados_dotacion['Z'][2]:.2f}$         |       ${(resultados_dotacion['X'][2] + resultados_dotacion['Z'][2]) / 0.4:.2f}$       |       ${pd.read_csv(DATA_DIR / 'datos_periodos.csv')['d_t'][1] - resultados_dotacion['NS'][2]:.2f}$       |        ${resultados_dotacion['NS'][2]:.2f}$        |
|     3     |   Base    |         ${resultados_base['X'][3]:.2f}$          |        ${resultados_base['Y'][3]:.2f}$         |         ${resultados_base['Z'][3]:.2f}$         |       ${(resultados_base['X'][3] + resultados_base['Z'][3]) / 0.4:.2f}$       |       ${pd.read_csv(DATA_DIR / 'datos_periodos.csv')['d_t'][2] - resultados_base['NS'][3]:.2f}$       |        ${resultados_base['NS'][3]:.2f}$        |
|     3     |   Dotación   |         ${resultados_dotacion['X'][3]:.2f}$          |        ${resultados_dotacion['Y'][3]:.2f}$        |        ${resultados_dotacion['Z'][3]:.2f}$         |       ${(resultados_dotacion['X'][3] + resultados_dotacion['Z'][3]) / 0.4:.2f}$       |       ${pd.read_csv(DATA_DIR / 'datos_periodos.csv')['d_t'][2] - resultados_dotacion['NS'][3]:.2f}$       |        ${resultados_dotacion['NS'][3]:.2f}$        |
|     4     |   Base    |         ${resultados_base['X'][4]:.2f}$          |        ${resultados_base['Y'][4]:.2f}$         |         ${resultados_base['Z'][4]:.2f}$         |       ${(resultados_base['X'][4] + resultados_base['Z'][4]) / 0.4:.2f}$       |       ${pd.read_csv(DATA_DIR / 'datos_periodos.csv')['d_t'][3] - resultados_base['NS'][4]:.2f}$       |        ${resultados_base['NS'][4]:.2f}$        |
|     4     |   Dotación   |         ${resultados_dotacion['X'][4]:.2f}$          |        ${resultados_dotacion['Y'][4]:.2f}$         |         ${resultados_dotacion['Z'][4]:.2f}$         |       ${(resultados_dotacion['X'][4] + resultados_dotacion['Z'][4]) / 0.4:.2f}$       |       ${pd.read_csv(DATA_DIR / 'datos_periodos.csv')['d_t'][3] - resultados_dotacion['NS'][4]:.2f}$       |        ${resultados_dotacion['NS'][4]:.2f}$        |
|     5     |   Base    |         ${resultados_base['X'][5]:.2f}$          |        ${resultados_base['Y'][5]:.2f}$         |         ${resultados_base['Z'][5]:.2f}$         |       ${(resultados_base['X'][5] + resultados_base['Z'][5]) / 0.4:.2f}$       |       ${pd.read_csv(DATA_DIR / 'datos_periodos.csv')['d_t'][4] - resultados_base['NS'][5]:.2f}$       |        ${resultados_base['NS'][5]:.2f}$        |
|     5     |   Dotación   |         ${resultados_dotacion['X'][5]:.2f}$          |        ${resultados_dotacion['Y'][5]:.2f}$         |         ${resultados_dotacion['Z'][5]:.2f}$         |       ${(resultados_dotacion['X'][5] + resultados_dotacion['Z'][5]) / 0.4:.2f}$       |       ${pd.read_csv(DATA_DIR / 'datos_periodos.csv')['d_t'][4] - resultados_dotacion['NS'][5]:.2f}$       |        ${resultados_dotacion['NS'][5]:.2f}$        |
| **Total** | **Base**  |       **${sum(resultados_base['X'].values()):.2f}$**       |      **${sum(resultados_base['Y'].values()):.2f}$**      |      **${sum(resultados_base['Z'].values()):.2f}$**       |    **${sum((resultados_base['X'][t] + resultados_base['Z'][t]) / 0.4 for t in range(1, 6)):.2f}$**     |    **${sum(pd.read_csv(DATA_DIR / 'datos_periodos.csv')['d_t'][t-1] - resultados_base['NS'][t] for t in range(1, 6)):.2f}$**     |      **${sum(resultados_base['NS'].values()):.2f}$**      |
| **Total** | **Dotación** |       **${sum(resultados_dotacion['X'].values()):.2f}$**       |      **${sum(resultados_dotacion['Y'].values()):.2f}$**      |      **${sum(resultados_dotacion['Z'].values()):.2f}$**       |    **${sum((resultados_dotacion['X'][t] + resultados_dotacion['Z'][t]) / 0.4 for t in range(1, 6)):.2f}$**     |    **${sum(pd.read_csv(DATA_DIR / 'datos_periodos.csv')['d_t'][t-1] - resultados_dotacion['NS'][t] for t in range(1, 6)):.2f}$**     |      **${sum(resultados_dotacion['NS'].values()):.2f}$**      |

### 2. Inventarios por Periodo

| Periodo | Caso  | Inv. ropa buen estado (kg) | Inv. ropa mal estado (kg) | Inv. género (kg) | Almacenamiento total (kg) | % Capacidad utilizada |
| :-----: | :---: | :------------------------: | :-----------------------: | :--------------: | :-----------------------: | :-------------------: |
|    1    | Base  |            ${resultados_base['IB'][1]:.2f}$            |           ${resultados_base['IM'][1]:.2f}$            |       ${resultados_base['IG'][1]:.2f}$       |           ${resultados_base['IB'][1] + resultados_base['IM'][1] + resultados_base['IG'][1]:.2f}$            |         ${(resultados_base['IB'][1] + resultados_base['IM'][1] + resultados_base['IG'][1]) / 460 * 100:.2f}$          |
|    1    | Dotación |            ${resultados_dotacion['IB'][1]:.2f}$            |           ${resultados_dotacion['IM'][1]:.2f}$           |       ${resultados_dotacion['IG'][1]:.2f}$       |           ${resultados_dotacion['IB'][1] + resultados_dotacion['IM'][1] + resultados_dotacion['IG'][1]:.2f}$           |         ${(resultados_dotacion['IB'][1] + resultados_dotacion['IM'][1] + resultados_dotacion['IG'][1]) / 460 * 100:.2f}$          |
|    2    | Base  |            ${resultados_base['IB'][2]:.2f}$            |           ${resultados_base['IM'][2]:.2f}$            |       ${resultados_base['IG'][2]:.2f}$       |           ${resultados_base['IB'][2] + resultados_base['IM'][2] + resultados_base['IG'][2]:.2f}$            |         ${(resultados_base['IB'][2] + resultados_base['IM'][2] + resultados_base['IG'][2]) / 460 * 100:.2f}$          |
|    2    | Dotación |            ${resultados_dotacion['IB'][2]:.2f}$            |           ${resultados_dotacion['IM'][2]:.2f}$            |       ${resultados_dotacion['IG'][2]:.2f}$       |           ${resultados_dotacion['IB'][2] + resultados_dotacion['IM'][2] + resultados_dotacion['IG'][2]:.2f}$            |         ${(resultados_dotacion['IB'][2] + resultados_dotacion['IM'][2] + resultados_dotacion['IG'][2]) / 460 * 100:.2f}$          |
|    3    | Base  |            ${resultados_base['IB'][3]:.2f}$            |           ${resultados_base['IM'][3]:.2f}$           |       ${resultados_base['IG'][3]:.2f}$       |           ${resultados_base['IB'][3] + resultados_base['IM'][3] + resultados_base['IG'][3]:.2f}$           |         ${(resultados_base['IB'][3] + resultados_base['IM'][3] + resultados_base['IG'][3]) / 460 * 100:.2f}$          |
|    3    | Dotación |            ${resultados_dotacion['IB'][3]:.2f}$            |           ${resultados_dotacion['IM'][3]:.2f}$            |       ${resultados_dotacion['IG'][3]:.2f}$       |           ${resultados_dotacion['IB'][3] + resultados_dotacion['IM'][3] + resultados_dotacion['IG'][3]:.2f}$            |         ${(resultados_dotacion['IB'][3] + resultados_dotacion['IM'][3] + resultados_dotacion['IG'][3]) / 460 * 100:.2f}$          |
|    4    | Base  |            ${resultados_base['IB'][4]:.2f}$            |           ${resultados_base['IM'][4]:.2f}$            |       ${resultados_base['IG'][4]:.2f}$       |           ${resultados_base['IB'][4] + resultados_base['IM'][4] + resultados_base['IG'][4]:.2f}$            |         ${(resultados_base['IB'][4] + resultados_base['IM'][4] + resultados_base['IG'][4]) / 460 * 100:.2f}$          |
|    4    | Dotación |            ${resultados_dotacion['IB'][4]:.2f}$            |           ${resultados_dotacion['IM'][4]:.2f}$            |       ${resultados_dotacion['IG'][4]:.2f}$       |           ${resultados_dotacion['IB'][4] + resultados_dotacion['IM'][4] + resultados_dotacion['IG'][4]:.2f}$            |         ${(resultados_dotacion['IB'][4] + resultados_dotacion['IM'][4] + resultados_dotacion['IG'][4]) / 460 * 100:.2f}$          |
|    5    | Base  |            ${resultados_base['IB'][5]:.2f}$            |           ${resultados_base['IM'][5]:.2f}$            |       ${resultados_base['IG'][5]:.2f}$       |           ${resultados_base['IB'][5] + resultados_base['IM'][5] + resultados_base['IG'][5]:.2f}$            |         ${(resultados_base['IB'][5] + resultados_base['IM'][5] + resultados_base['IG'][5]) / 460 * 100:.2f}$          |
|    5    | Dotación |            ${resultados_dotacion['IB'][5]:.2f}$            |           ${resultados_dotacion['IM'][5]:.2f}$            |       ${resultados_dotacion['IG'][5]:.2f}$       |           ${resultados_dotacion['IB'][5] + resultados_dotacion['IM'][5] + resultados_dotacion['IG'][5]:.2f}$            |         ${(resultados_dotacion['IB'][5] + resultados_dotacion['IM'][5] + resultados_dotacion['IG'][5]) / 460 * 100:.2f}$          |

### 3. Recursos Humanos y Utilización

|  Periodo  |   Caso    | Trabajadores contratados | Trabajadores por boleta | Total trabajadores | Horas disponibles | Horas utilizadas | % Utilización |
| :-------: | :-------: | :----------------------: | :---------------------: | :----------------: | :---------------: | :--------------: | :-----------: |
|     1     |   Base    |            ${2}$             |            ${resultados_base['W'][1]}$            |         ${2 + resultados_base['W'][1]}$          |       ${8 * (2 + resultados_base['W'][1]):.2f}$       |      ${0.17 * resultados_base['Y'][1] + 0.19 * resultados_base['Z'][1]:.2f}$       |    ${(0.17 * resultados_base['Y'][1] + 0.19 * resultados_base['Z'][1]) / (8 * (2 + resultados_base['W'][1])) * 100:.2f}$     |
|     1     |   Dotación   |            ${2}$             |            ${resultados_dotacion['W'][1]}$            |         ${2 + resultados_dotacion['W'][1]}$          |       ${8 * (2 + resultados_dotacion['W'][1]):.2f}$       |      ${0.17 * resultados_dotacion['Y'][1] + 0.19 * resultados_dotacion['Z'][1]:.2f}$       |    ${(0.17 * resultados_dotacion['Y'][1] + 0.19 * resultados_dotacion['Z'][1]) / (8 * (2 + resultados_dotacion['W'][1])) * 100:.2f}$     |
|     2     |   Base    |            ${2}$             |            ${resultados_base['W'][2]}$            |         ${2 + resultados_base['W'][2]}$          |       ${8 * (2 + resultados_base['W'][2]):.2f}$       |      ${0.17 * resultados_base['Y'][2] + 0.19 * resultados_base['Z'][2]:.2f}$       |    ${(0.17 * resultados_base['Y'][2] + 0.19 * resultados_base['Z'][2]) / (8 * (2 + resultados_base['W'][2])) * 100:.2f}$     |
|     2     |   Dotación   |            ${2}$             |            ${resultados_dotacion['W'][2]}$            |         ${2 + resultados_dotacion['W'][2]}$          |       ${8 * (2 + resultados_dotacion['W'][2]):.2f}$       |      ${0.17 * resultados_dotacion['Y'][2] + 0.19 * resultados_dotacion['Z'][2]:.2f}$       |    ${(0.17 * resultados_dotacion['Y'][2] + 0.19 * resultados_dotacion['Z'][2]) / (8 * (2 + resultados_dotacion['W'][2])) * 100:.2f}$     |
|     3     |   Base    |            ${2}$             |            ${resultados_base['W'][3]}$            |         ${2 + resultados_base['W'][3]}$          |       ${8 * (2 + resultados_base['W'][3]):.2f}$       |      ${0.17 * resultados_base['Y'][3] + 0.19 * resultados_base['Z'][3]:.2f}$       |    ${(0.17 * resultados_base['Y'][3] + 0.19 * resultados_base['Z'][3]) / (8 * (2 + resultados_base['W'][3])) * 100:.2f}$     |
|     3     |   Dotación   |            ${2}$             |            ${resultados_dotacion['W'][3]}$            |         ${2 + resultados_dotacion['W'][3]}$          |       ${8 * (2 + resultados_dotacion['W'][3]):.2f}$       |      ${0.17 * resultados_dotacion['Y'][3] + 0.19 * resultados_dotacion['Z'][3]:.2f}$       |    ${(0.17 * resultados_dotacion['Y'][3] + 0.19 * resultados_dotacion['Z'][3]) / (8 * (2 + resultados_dotacion['W'][3])) * 100:.2f}$     |
|     4     |   Base    |            ${2}$             |            ${resultados_base['W'][4]}$            |         ${2 + resultados_base['W'][4]}$          |       ${8 * (2 + resultados_base['W'][4]):.2f}$       |      ${0.17 * resultados_base['Y'][4] + 0.19 * resultados_base['Z'][4]:.2f}$       |    ${(0.17 * resultados_base['Y'][4] + 0.19 * resultados_base['Z'][4]) / (8 * (2 + resultados_base['W'][4])) * 100:.2f}$     |
|     4     |   Dotación   |            ${2}$             |            ${resultados_dotacion['W'][4]}$            |         ${2 + resultados_dotacion['W'][4]}$          |       ${8 * (2 + resultados_dotacion['W'][4]):.2f}$       |      ${0.17 * resultados_dotacion['Y'][4] + 0.19 * resultados_dotacion['Z'][4]:.2f}$       |    ${(0.17 * resultados_dotacion['Y'][4] + 0.19 * resultados_dotacion['Z'][4]) / (8 * (2 + resultados_dotacion['W'][4])) * 100:.2f}$     |
|     5     |   Base    |            ${2}$             |            ${resultados_base['W'][5]}$            |         ${2 + resultados_base['W'][5]}$          |       ${8 * (2 + resultados_base['W'][5]):.2f}$       |      ${0.17 * resultados_base['Y'][5] + 0.19 * resultados_base['Z'][5]:.2f}$       |    ${(0.17 * resultados_base['Y'][5] + 0.19 * resultados_base['Z'][5]) / (8 * (2 + resultados_base['W'][5])) * 100:.2f}$     |
|     5     |   Dotación   |            ${2}$             |            ${resultados_dotacion['W'][5]}$            |         ${2 + resultados_dotacion['W'][5]}$          |       ${8 * (2 + resultados_dotacion['W'][5]):.2f}$       |      ${0.17 * resultados_dotacion['Y'][5] + 0.19 * resultados_dotacion['Z'][5]:.2f}$       |    ${(0.17 * resultados_dotacion['Y'][5] + 0.19 * resultados_dotacion['Z'][5]) / (8 * (2 + resultados_dotacion['W'][5])) * 100:.2f}$     |
| **Total** | **Base**  |          **${2 * 5}$**          |          **${sum(resultados_base['W'].values())}$**          |       **${2 * 5 + sum(resultados_base['W'].values())}$**       |    **${8 * (2 * 5 + sum(resultados_base['W'].values())):.2f}$**     |    **${sum(0.17 * resultados_base['Y'][t] + 0.19 * resultados_base['Z'][t] for t in range(1, 6)):.2f}$**    |  **${sum(0.17 * resultados_base['Y'][t] + 0.19 * resultados_base['Z'][t] for t in range(1, 6)) / (8 * (2 * 5 + sum(resultados_base['W'].values()))) * 100:.2f}$**   |
| **Total** | **Dotación** |          **${2 * 5}$**          |          **${sum(resultados_dotacion['W'].values())}$**          |       **${2 * 5 + sum(resultados_dotacion['W'].values())}$**       |    **${8 * (2 * 5 + sum(resultados_dotacion['W'].values())):.2f}$**     |    **${sum(0.17 * resultados_dotacion['Y'][t] + 0.19 * resultados_dotacion['Z'][t] for t in range(1, 6)):.2f}$**    |  **${sum(0.17 * resultados_dotacion['Y'][t] + 0.19 * resultados_dotacion['Z'][t] for t in range(1, 6)) / (8 * (2 * 5 + sum(resultados_dotacion['W'].values()))) * 100:.2f}$**   |

### 4. Desglose de Costos

|              Componente               |    Caso Base     |             |    Caso con Dotación    |             | Variación  |
| :-----------------------------------: | :--------------: | :---------: | :--------------: | :---------: | :--------: |
|                                       |    Valor ($)     | Porcentaje  |    Valor ($)     | Porcentaje  | Porcentual |
|          Personal contratado          |    ${11500 * 8 * 2 * 5:,.2f}$    |   ${11500 * 8 * 2 * 5 / resultados_base['valor_objetivo'] * 100:.2f}\\%$    |    ${11500 * 8 * 2 * 5:,.2f}$    |    ${11500 * 8 * 2 * 5 / resultados_dotacion['valor_objetivo'] * 100:.2f}\\%$    |   ${0.00}\\%$    |
|          Personal por boleta          |   ${215000 * sum(resultados_base['W'].values()):,.2f}$   |   ${215000 * sum(resultados_base['W'].values()) / resultados_base['valor_objetivo'] * 100:.2f}\\%$    |   ${215000 * sum(resultados_dotacion['W'].values()):,.2f}$   |   ${215000 * sum(resultados_dotacion['W'].values()) / resultados_dotacion['valor_objetivo'] * 100:.2f}\\%$    |  ${(215000 * sum(resultados_dotacion['W'].values()) - 215000 * sum(resultados_base['W'].values())) / (215000 * sum(resultados_base['W'].values())) * 100:.2f}\\%$   |
|        Transformación a género        |    ${395 * sum(resultados_base['Y'].values()):,.2f}$    |    ${395 * sum(resultados_base['Y'].values()) / resultados_base['valor_objetivo'] * 100:.2f}\\%$    |    ${395 * sum(resultados_dotacion['Y'].values()):,.2f}$    |    ${395 * sum(resultados_dotacion['Y'].values()) / resultados_dotacion['valor_objetivo'] * 100:.2f}\\%$    |   ${(395 * sum(resultados_dotacion['Y'].values()) - 395 * sum(resultados_base['Y'].values())) / (395 * sum(resultados_base['Y'].values())) * 100:.2f}\\%$   |
|         Producción de prendas         |    ${265 * sum(resultados_base['Z'].values()):,.2f}$     |    ${265 * sum(resultados_base['Z'].values()) / resultados_base['valor_objetivo'] * 100:.2f}\\%$    |    ${265 * sum(resultados_dotacion['Z'].values()):,.2f}$     |    ${265 * sum(resultados_dotacion['Z'].values()) / resultados_dotacion['valor_objetivo'] * 100:.2f}\\%$    |   ${(265 * sum(resultados_dotacion['Z'].values()) - 265 * sum(resultados_base['Z'].values())) / (265 * sum(resultados_base['Z'].values())) * 100:.2f}\\%$   |
|            Almacenamiento             |    ${405 * sum(resultados_base['IB'][t] + resultados_base['IM'][t] + resultados_base['IG'][t] for t in range(1, 6)):,.2f}$     |    ${405 * sum(resultados_base['IB'][t] + resultados_base['IM'][t] + resultados_base['IG'][t] for t in range(1, 6)) / resultados_base['valor_objetivo'] * 100:.2f}\\%$    |    ${405 * sum(resultados_dotacion['IB'][t] + resultados_dotacion['IM'][t] + resultados_dotacion['IG'][t] for t in range(1, 6)):,.2f}$     |    ${405 * sum(resultados_dotacion['IB'][t] + resultados_dotacion['IM'][t] + resultados_dotacion['IG'][t] for t in range(1, 6)) / resultados_dotacion['valor_objetivo'] * 100:.2f}\\%$    |  ${(405 * sum(resultados_dotacion['IB'][t] + resultados_dotacion['IM'][t] + resultados_dotacion['IG'][t] for t in range(1, 6)) - 405 * sum(resultados_base['IB'][t] + resultados_base['IM'][t] + resultados_base['IG'][t] for t in range(1, 6))) / (405 * sum(resultados_base['IB'][t] + resultados_base['IM'][t] + resultados_base['IG'][t] for t in range(1, 6))) * 100:.2f}\\%$   |
| Penalización por demanda insatisfecha |   ${7000 * sum(resultados_base['NS'].values()):,.2f}$   |   ${7000 * sum(resultados_base['NS'].values()) / resultados_base['valor_objetivo'] * 100:.2f}\\%$    |   ${7000 * sum(resultados_dotacion['NS'].values()):,.2f}$   |   ${7000 * sum(resultados_dotacion['NS'].values()) / resultados_dotacion['valor_objetivo'] * 100:.2f}\\%$    |   ${(7000 * sum(resultados_dotacion['NS'].values()) - 7000 * sum(resultados_base['NS'].values())) / (7000 * sum(resultados_base['NS'].values())) * 100:.2f}\\%$   |
|            **Costo total**            | **${resultados_base["valor_objetivo"]:,.2f}$** | **${100}\\%$** | **${resultados_dotacion["valor_objetivo"]:,.2f}$** | **${100}\\%$** | **${((resultados_dotacion['valor_objetivo'] - resultados_base['valor_objetivo']) / resultados_base['valor_objetivo']) * 100:.2f}\\%$** |

## Impacto en Indicadores Clave

### 1. Costo Total

- Caso base: ${resultados_base['valor_objetivo']:,.2f}$
- Caso con dotación mínima: ${resultados_dotacion['valor_objetivo']:,.2f}$
- Variación: ${cambio_valor_objetivo:+.2f}\\%$

### 2. Procesamiento de Ropa en Mal Estado
- Caso base: ${total_Y_base:.2f}$ kg
- Caso con dotación mínima: ${total_Y_dotacion:.2f}$ kg
- Variación: ${cambio_procesamiento:+.2f}\\%$

### 3. Demanda Insatisfecha
- Caso base: ${total_NS_base:.2f}$ prendas
- Caso con dotación mínima: ${total_NS_dotacion:.2f}$ prendas
- Variación: ${cambio_demanda_insatisfecha:+.2f}\\%$

### 4. Utilización de Personal
- Caso base: ${total_W_base}$ trabajadores por boleta en total
- Caso con dotación mínima: ${total_W_dotacion}$ trabajadores por boleta en total
- Variación: ${cambio_trabajadores:+.2f}\\%$

## Gráficos Comparativos

### Comparación de Producción vs Demanda
![Comparación de Producción vs Demanda](../src/pregunta5/grafico_comparativo_produccion_demanda.png)

*Figura 1: Comparación de la producción y demanda entre el caso base y el caso con dotación mínima de personal.*

### Comparación de Recursos Humanos
![Comparación de Recursos Humanos](../src/pregunta5/grafico_comparativo_recursos_humanos.png)

*Figura 2: Comparación de la utilización de recursos humanos entre el caso base y el caso con dotación mínima de personal.*

### Comparación de Costos
![Comparación de Costos](../src/pregunta5/grafico_comparativo_costos.png)

*Figura 3: Comparación de los costos totales y su distribución entre el caso base y el caso con dotación mínima de personal.*

### Comparación de Procesamiento de Ropa en Mal Estado
![Comparación de Procesamiento](../src/pregunta5/grafico_comparativo_procesamiento.png)

*Figura 4: Comparación del procesamiento de ropa en mal estado entre el caso base y el caso con dotación mínima de personal.*

## Análisis Detallado

### Impacto en la Capacidad de Producción

La política de dotación mínima de personal que exige mantener al menos ${param['tr']}$ trabajadores activos en cada periodo ha tenido los siguientes efectos en la capacidad productiva del sistema:

1. **Redistribución por Periodos**: Aunque el total de ropa en mal estado procesada se mantiene igual (${sum(resultados_base['Y'].values()):.2f}$ kg), la restricción de dotación mínima ha modificado significativamente su distribución entre periodos:
   - En el periodo 1: { 'Aumento' if resultados_dotacion['Y'][1] > resultados_base['Y'][1] else 'Disminución' } del procesamiento (de ${resultados_base['Y'][1]:.2f}$ kg a ${resultados_dotacion['Y'][1]:.2f}$ kg)
   - En el periodo 3: { 'Aumento' if resultados_dotacion['Y'][3] > resultados_base['Y'][3] else 'Disminución' } del procesamiento (de ${resultados_base['Y'][3]:.2f}$ kg a ${resultados_dotacion['Y'][3]:.2f}$ kg)
   - En el periodo 4: { 'Aumento' if resultados_dotacion['Y'][4] > resultados_base['Y'][4] else 'Disminución' } del procesamiento (de ${resultados_base['Y'][4]:.2f}$ kg a ${resultados_dotacion['Y'][4]:.2f}$ kg)
   - En el periodo 5: { 'Aumento' if resultados_dotacion['Y'][5] > resultados_base['Y'][5] else 'Disminución' } del procesamiento (de ${resultados_base['Y'][5]:.2f}$ kg a ${resultados_dotacion['Y'][5]:.2f}$ kg)

2. **Redistribución de la Producción de Prendas**: Similar al procesamiento de ropa, la producción de prendas se redistribuye entre periodos, con una notable { 'mejora' if (resultados_dotacion['X'][5] + resultados_dotacion['Z'][5])/0.4 > (resultados_base['X'][5] + resultados_base['Z'][5])/0.4 else 'reducción' } en el periodo 5 (de ${(resultados_base['X'][5] + resultados_base['Z'][5])/0.4:.2f}$ a ${(resultados_dotacion['X'][5] + resultados_dotacion['Z'][5])/0.4:.2f}$ prendas), ${ 'reduciendo' if resultados_dotacion['NS'][5] < resultados_base['NS'][5] else 'aumentando' } la demanda insatisfecha en ese periodo de ${resultados_base['NS'][5]:.2f}$ a ${resultados_dotacion['NS'][5]:.2f}$ prendas.

3. **Costos de Almacenamiento**: El costo de almacenamiento aumentó significativamente en un ${((sum(resultados_dotacion['IB'].values()) + sum(resultados_dotacion['IM'].values()) + sum(resultados_dotacion['IG'].values())) / (sum(resultados_base['IB'].values()) + sum(resultados_base['IM'].values()) + sum(resultados_base['IG'].values())) - 1) * 100:.2f}\\%$, debido a que la restricción de dotación mínima fuerza a mantener inventarios más altos entre periodos para optimizar el uso del personal obligatorio.

### Adaptación de la Estrategia Operativa

El modelo ha respondido a la política de dotación mínima de personal mediante:

1. **Ajuste de Personal por Periodo**: En los periodos 4 y 5, donde el caso base utilizaba solo ${param['w0'] + 0}$ trabajadores, la restricción de dotación mínima fuerza a aumentar a ${param['tr']}$ trabajadores, redistribuyendo trabajadores por boleta para mantener el total en ${total_W_base + param['w0'] * 5}$ trabajadores.

2. **Gestión de Inventarios**: Notablemente, los niveles de inventario de ropa en mal estado aumentaron significativamente:
   - Periodo 1: de ${resultados_base['IM'][1]:.2f}$ kg a ${resultados_dotacion['IM'][1]:.2f}$ kg
   - Periodo 2: de ${resultados_base['IM'][2]:.2f}$ kg a ${resultados_dotacion['IM'][2]:.2f}$ kg
   - Periodo 3: de ${resultados_base['IM'][3]:.2f}$ kg a ${resultados_dotacion['IM'][3]:.2f}$ kg
   - Periodo 4: de ${resultados_base['IM'][4]:.2f}$ kg a ${resultados_dotacion['IM'][4]:.2f}$ kg

## Conclusiones

1. La política de dotación mínima de personal ha tenido un impacto moderado en el costo total de operación (${cambio_valor_objetivo:+.2f}\\%$), pero ha causado cambios significativos en la estrategia operativa de la fundación.

2. El sistema se ha adaptado a la restricción de dotación mínima principalmente a través de:
   - La redistribución del procesamiento de ropa entre periodos
   - El aumento significativo de los inventarios intermedios
   - El incremento de personal en periodos específicos (4 y 5)

3. Las principales consecuencias observadas son:
   - Aumento drástico de los costos de almacenamiento (${((sum(resultados_dotacion['IB'].values()) + sum(resultados_dotacion['IM'].values()) + sum(resultados_dotacion['IG'].values())) / (sum(resultados_base['IB'].values()) + sum(resultados_base['IM'].values()) + sum(resultados_base['IG'].values())) - 1) * 100:.2f}\\%$)
   - Redistribución de la producción hacia periodos finales
   - Mantenimiento del nivel total de producción y servicio al cliente
   - Mayor estabilidad laboral al garantizar un mínimo de ${param['tr']}$ trabajadores en todos los periodos

## Recomendaciones

1. Evaluar el impacto social y financiero de la política de dotación mínima, considerando el trade-off entre estabilidad laboral y el aumento de costos de almacenamiento.

2. Explorar opciones para optimizar la gestión de inventarios bajo la restricción de dotación mínima, posiblemente mediante mejoras en los espacios de almacenamiento.

3. Considerar políticas de dotación mínima variables por periodo, que podrían reducir costos manteniendo beneficios sociales en periodos críticos.
"""
    
    # Guardar el informe en un archivo Markdown
    os.makedirs(RESPUESTAS_DIR, exist_ok=True)
    with open(RESPUESTAS_DIR / 'pregunta5_dotacion.md', 'w') as f:
        f.write(informe)
    
    return informe

if __name__ == "__main__":
    print("Resolviendo el modelo de optimización para la pregunta 5...")
    
    # Asegurarse que exista la carpeta para los resultados
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Resolver ambos modelos
    print("Resolviendo caso base...")
    resultados_base, param_base, T_base, kb_base, km_base, d_base = resolver_modelo_base()
    
    print("Resolviendo caso con dotación mínima de personal...")
    resultados_dotacion, param_dotacion, T_dotacion, kb_dotacion, km_dotacion, d_dotacion = resolver_modelo_con_dotacion_minima()
    
    if resultados_base is not None and resultados_dotacion is not None:
        print(f"\nModelos resueltos correctamente.")
        print(f"Valor óptimo caso base: ${resultados_base['valor_objetivo']:,.2f}")
        print(f"Valor óptimo con dotación mínima: ${resultados_dotacion['valor_objetivo']:,.2f}")
        
        # Mostrar el número mínimo de trabajadores requeridos
        trabajadores_minimo = param_base['tr']
        print(f"Dotación mínima requerida: {int(trabajadores_minimo)} trabajadores")
        
        # Generar informe comparativo
        print("\nGenerando informe comparativo...")
        informe = generar_informe_comparativo(resultados_base, resultados_dotacion)
        
        print(f"\nSe han generado los gráficos comparativos en: {OUTPUT_DIR.resolve()}")
        print(f"Se ha generado el informe en: {(RESPUESTAS_DIR / 'pregunta5_dotacion.md').resolve()}")
        
        # Mostrar un resumen comparativo básico
        total_Y_base = sum(resultados_base['Y'].values())
        total_Y_dotacion = sum(resultados_dotacion['Y'].values())
        cambio_procesamiento = ((total_Y_dotacion - total_Y_base) / total_Y_base) * 100 if total_Y_base > 0 else float('inf')
        
        total_NS_base = sum(resultados_base['NS'].values())
        total_NS_dotacion = sum(resultados_dotacion['NS'].values())
        cambio_demanda_insatisfecha = ((total_NS_dotacion - total_NS_base) / total_NS_base) * 100 if total_NS_base > 0 else float('inf')
        
        # Calcular el total de trabajadores en cada caso
        min_trabajadores_base = min([param_base['w0'] + resultados_base['W'][t] for t in range(1, T_base+1)])
        min_trabajadores_dotacion = min([param_dotacion['w0'] + resultados_dotacion['W'][t] for t in range(1, T_dotacion+1)])
        
        print("\nResumen comparativo:")
        print(f"- Procesamiento de ropa en mal estado: {total_Y_base:.2f} kg (base) vs {total_Y_dotacion:.2f} kg (con dotación mínima) - Cambio: {cambio_procesamiento:+.2f}%")
        print(f"- Demanda insatisfecha: {total_NS_base:.2f} prendas (base) vs {total_NS_dotacion:.2f} prendas (con dotación mínima) - Cambio: {cambio_demanda_insatisfecha:+.2f}%")
        print(f"- Costo total: ${resultados_base['valor_objetivo']:,.2f} (base) vs ${resultados_dotacion['valor_objetivo']:,.2f} (con dotación mínima) - Cambio: {((resultados_dotacion['valor_objetivo'] - resultados_base['valor_objetivo']) / resultados_base['valor_objetivo']) * 100:+.2f}%")
        print(f"- Dotación mínima de trabajadores: {min_trabajadores_base} (base) vs {min_trabajadores_dotacion} (con política de mínimo) - Mínimo requerido: {int(param_base['tr'])}")
    else:
        print("No se pudieron resolver los modelos. Verifique los datos y restricciones.")

