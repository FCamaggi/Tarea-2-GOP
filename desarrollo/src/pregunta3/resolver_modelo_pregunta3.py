#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Implementación del modelo de optimización para la Fundación Circular.
Este script resuelve el modelo para la pregunta 3, considerando una falla técnica
que aumenta en un 25% el tiempo requerido para procesar cada kilogramo de ropa en mal estado.
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
OUTPUT_DIR = BASE_DIR / "desarrollo" / "src" / "pregunta3"
RESPUESTAS_DIR = BASE_DIR / "desarrollo" / "respuestas"

def resolver_modelo_general(tau_g_override=None):
    """
    Resuelve el modelo de optimización con los parámetros dados.
    Si se proporciona tau_g_override, usa ese valor en lugar del valor por defecto.
    """
    # Cargar datos
    parametros = pd.read_csv(DATA_DIR / 'datos_parametros.csv')
    periodos = pd.read_csv(DATA_DIR / 'datos_periodos.csv')
    
    # Convertir los parámetros a un diccionario para fácil acceso
    param_dict = dict(zip(parametros['Parametro'], parametros['Valor']))
    
    # Si se proporciona un valor específico para tau_g, usarlo
    if tau_g_override is not None:
        param_dict['tau_g'] = tau_g_override
    
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
    
    return resultados, param_dict, T, kb, km, d

def resolver_modelo_base():
    """Resuelve el modelo de optimización lineal base sin falla técnica"""
    parametros = pd.read_csv(DATA_DIR / 'datos_parametros.csv')
    param_dict = dict(zip(parametros['Parametro'], parametros['Valor']))
    tau_g_original = param_dict['tau_g']  # Obtener el valor original
    return resolver_modelo_general(tau_g_original)

def resolver_modelo_falla():
    """Resuelve el modelo de optimización considerando la falla técnica"""
    parametros = pd.read_csv(DATA_DIR / 'datos_parametros.csv')
    param_dict = dict(zip(parametros['Parametro'], parametros['Valor']))
    tau_g_con_falla = param_dict['tau_g'] * 1.25  # Aumentar en 25%
    return resolver_modelo_general(tau_g_con_falla)

def cargar_datos():
    """Carga los datos de los archivos CSV"""
    parametros = pd.read_csv(DATA_DIR / 'datos_parametros.csv')
    periodos = pd.read_csv(DATA_DIR / 'datos_periodos.csv')
    
    # Convertir los parámetros a un diccionario para fácil acceso
    param_dict = dict(zip(parametros['Parametro'], parametros['Valor']))
    
    # Aumentar en 25% el tiempo requerido para procesar ropa en mal estado
    param_dict['tau_g'] = param_dict['tau_g'] * 1.25
    
    # Extraer los datos por periodo
    T = len(periodos)  # Número de periodos
    kb = periodos['kb_t'].tolist()
    km = periodos['km_t'].tolist()
    d = periodos['d_t'].tolist()
    
    return param_dict, T, kb, km, d

def generar_graficos_comparativos(resultados_base, resultados_falla):
    """Genera gráficos comparativos entre el caso base y el caso con falla"""
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
    
    df_prod_falla = pd.DataFrame({
        'Periodo': periodos,
        'Prendas Producidas': [
            (resultados_falla['X'][t] + resultados_falla['Z'][t]) / param['p'] 
            for t in periodos
        ],
        'Demanda Satisfecha': [
            pd.read_csv(DATA_DIR / 'datos_periodos.csv')['d_t'][t-1] - resultados_falla['NS'][t] 
            for t in periodos
        ],
        'Demanda Insatisfecha': [resultados_falla['NS'][t] for t in periodos]
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
    
    df_rec_falla = pd.DataFrame({
        'Periodo': periodos,
        'Total Trabajadores': [param['w0'] + resultados_falla['W'][t] for t in periodos],
        'Trabajadores por Boleta': [resultados_falla['W'][t] for t in periodos],
        '% Utilización': [
            ((param['tau_g'] * 1.25) * resultados_falla['Y'][t] + param['tau_n'] * resultados_falla['Z'][t]) / 
            (param['h'] * (param['w0'] + resultados_falla['W'][t])) * 100 if (param['w0'] + resultados_falla['W'][t]) > 0 else 0
            for t in periodos
        ]
    })
    
    # Para procesamiento de ropa en mal estado
    df_proc_base = pd.DataFrame({
        'Periodo': periodos,
        'Ropa Mal Estado Procesada (kg)': [resultados_base['Y'][t] for t in periodos]
    })
    
    df_proc_falla = pd.DataFrame({
        'Periodo': periodos,
        'Ropa Mal Estado Procesada (kg)': [resultados_falla['Y'][t] for t in periodos]
    })
    
    # =====================================================
    # 1. Gráfico comparativo de Producción vs Demanda
    # =====================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), facecolor='white')
    fig.suptitle('Comparación de Producción y Demanda: Caso Base vs Caso con Falla', 
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
    
    # Gráfico para caso con falla
    ax2.set_facecolor('#f8f9fa')
    ax2.set_title('Caso con Falla Técnica', fontsize=14, fontweight='bold')
    
    # Crear gráfico apilado para la demanda (caso con falla)
    ax2.bar(df_prod_falla['Periodo'], df_prod_falla['Demanda Satisfecha'], 
            label='Demanda Satisfecha', alpha=0.8, color=paleta_colores['verde_exito'])
    ax2.bar(df_prod_falla['Periodo'], df_prod_falla['Demanda Insatisfecha'], 
            bottom=df_prod_falla['Demanda Satisfecha'], label='Demanda Insatisfecha', 
            alpha=0.8, color=paleta_colores['rojo_error'])
    
    # Añadir línea para producción (caso con falla)
    ax2.plot(df_prod_falla['Periodo'], df_prod_falla['Prendas Producidas'], 'o-', 
            color=paleta_colores['azul_principal'], linewidth=2.5, label='Prendas Producidas', 
            markersize=8, path_effects=[patheffects.withStroke(linewidth=4, foreground='white')])
    
    # Etiquetas y configuración (caso con falla)
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
    fig.suptitle('Comparación de Recursos Humanos: Caso Base vs Caso con Falla', 
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
    
    # Gráfico para caso con falla
    ax2.set_facecolor('#f8f9fa')
    ax2.set_title('Caso con Falla Técnica', fontsize=14, fontweight='bold')
    
    # Crear gráfico de barras para trabajadores (caso con falla)
    ax2.bar(df_rec_falla['Periodo'], [param['w0']] * len(periodos), width,
            label='Trabajadores Contratados', color=paleta_colores['azul_principal'])
    ax2.bar(df_rec_falla['Periodo'], df_rec_falla['Trabajadores por Boleta'], width,
            bottom=[param['w0']] * len(periodos), label='Trabajadores por Boleta',
            color=paleta_colores['naranja_alerta'])
    
    # Añadir etiquetas con el total de trabajadores (caso con falla)
    for i, periodo in enumerate(periodos):
        ax2.annotate(f'Total: {df_rec_falla["Total Trabajadores"].iloc[i]}',
                    xy=(periodo, df_rec_falla['Total Trabajadores'].iloc[i] + 0.3),
                    ha='center', va='bottom', fontweight='bold')
    
    # Añadir línea para porcentaje de utilización (caso con falla)
    ax4 = ax2.twinx()
    ax4.plot(df_rec_falla['Periodo'], df_rec_falla['% Utilización'], 'o-', 
            color=paleta_colores['verde_exito'], linewidth=2.5, label='% Utilización')
    
    # Configuración de ejes y etiquetas (caso con falla)
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
    
    costos_falla = {
        'Personal': param['cc'] * param['h'] * param['w0'] * 5 + param['ct'] * sum(resultados_falla['W'].values()),
        'Transformación': param['g'] * sum(resultados_falla['Y'].values()),
        'Producción': param['n'] * sum(resultados_falla['Z'].values()),
        'Almacenamiento': param['a'] * sum(sum(resultados_falla[inv].values()) for inv in ['IB', 'IM', 'IG']),
        'Penalización': param['cp'] * sum(resultados_falla['NS'].values())
    }
    
    categorias = list(costos_base.keys())
    valores_base = list(costos_base.values())
    valores_falla = list(costos_falla.values())
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), facecolor='white')
    fig.suptitle('Comparación de Costos: Caso Base vs Caso con Falla',
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
        labels=categorias,
        autopct='%1.1f%%',
        startangle=90,
        colors=plt.cm.tab10.colors[:len(categorias)],
        explode=[0.05 if i == valores_base.index(max(valores_base)) else 0 for i in range(len(valores_base))]
    )
    
    # Mejorar legibilidad de las etiquetas
    for autotext in autotexts:
        autotext.set_fontsize(10)
        autotext.set_fontweight('bold')
    
    # Gráfico de torta para caso con falla
    ax2.set_facecolor('#f8f9fa')
    ax2.set_title(f'Caso con Falla - Costo Total: ${resultados_falla["valor_objetivo"]:,.2f}',
                 fontsize=14, fontweight='bold')
    
    # Calcular porcentajes para las etiquetas
    total_falla = sum(valores_falla)
    porcentajes_falla = [100 * v / total_falla for v in valores_falla]
    
    # Gráfico de torta con etiquetas de porcentaje
    wedges, texts, autotexts = ax2.pie(
        valores_falla, 
        labels=categorias,
        autopct='%1.1f%%',
        startangle=90,
        colors=plt.cm.tab10.colors[:len(categorias)],
        explode=[0.05 if i == valores_falla.index(max(valores_falla)) else 0 for i in range(len(valores_falla))]
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
    
    # Crear barras para caso base y caso con falla
    bars1 = ax.bar(index - bar_width/2, df_proc_base['Ropa Mal Estado Procesada (kg)'],
                  bar_width, label='Caso Base', color=paleta_colores['azul_principal'])
    bars2 = ax.bar(index + bar_width/2, df_proc_falla['Ropa Mal Estado Procesada (kg)'],
                  bar_width, label='Caso con Falla', color=paleta_colores['rojo_error'])
    
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
    ax.axhline(y=sum(df_proc_falla['Ropa Mal Estado Procesada (kg)']/len(periodos)),
              color=paleta_colores['rojo_error'], linestyle='--', alpha=0.7,
              label=f'Promedio Falla: {sum(df_proc_falla["Ropa Mal Estado Procesada (kg)"]/len(periodos)):.1f} kg')
    
    # Actualizar leyenda con las líneas promedio
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels, loc='upper left')
    
    # Añadir cuadro de texto con la diferencia porcentual
    cambio_porcentual = ((sum(df_proc_falla['Ropa Mal Estado Procesada (kg)']) - 
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

def generar_informe_comparativo(resultados_base, resultados_falla):
    """Genera un informe comparativo entre el caso base y el caso con falla"""
    # Calcular diferencias en indicadores clave
    cambio_valor_objetivo = ((resultados_falla['valor_objetivo'] - resultados_base['valor_objetivo']) / 
                            resultados_base['valor_objetivo']) * 100
    
    total_Y_base = sum(resultados_base['Y'].values())
    total_Y_falla = sum(resultados_falla['Y'].values())
    cambio_procesamiento = ((total_Y_falla - total_Y_base) / total_Y_base) * 100 if total_Y_base > 0 else float('inf')
    
    total_NS_base = sum(resultados_base['NS'].values())
    total_NS_falla = sum(resultados_falla['NS'].values())
    cambio_demanda_insatisfecha = ((total_NS_falla - total_NS_base) / total_NS_base) * 100 if total_NS_base > 0 else float('inf')
    
    total_W_base = sum(resultados_base['W'].values())
    total_W_falla = sum(resultados_falla['W'].values())
    cambio_trabajadores = ((total_W_falla - total_W_base) / total_W_base) * 100 if total_W_base > 0 else float('inf')
    
    # Generar gráficos comparativos
    generar_graficos_comparativos(resultados_base, resultados_falla)
    
    # Crear el informe comparativo
    informe = f"""# Análisis Comparativo: Impacto de la Falla Técnica

## Introducción

Este análisis compara los resultados del modelo base con los resultados obtenidos tras la falla técnica que aumentó en un $25\\%$ el tiempo requerido para procesar cada kilogramo de ropa en mal estado.

## Tablas Comparativas

### 1. Planificación de Producción y Procesamiento

|  Periodo  |   Caso    | Ropa buen estado (kg) | Ropa mal estado (kg) | Género utilizado (kg) | Prendas producidas | Demanda satisfecha | Demanda insatisfecha |
| :-------: | :-------: | :-------------------: | :------------------: | :-------------------: | :----------------: | :----------------: | :------------------: |
|     1     |   Base    |         ${resultados_base['X'][1]:.2f}$         |        ${resultados_base['Y'][1]:.2f}$         |         ${resultados_base['Z'][1]:.2f}$         |       ${(resultados_base['X'][1] + resultados_base['Z'][1]) / 0.4:.2f}$       |       ${pd.read_csv(DATA_DIR / 'datos_periodos.csv')['d_t'][0] - resultados_base['NS'][1]:.2f}$       |        ${resultados_base['NS'][1]:.2f}$        |
|     1     |   Falla   |         ${resultados_falla['X'][1]:.2f}$         |        ${resultados_falla['Y'][1]:.2f}$         |         ${resultados_falla['Z'][1]:.2f}$         |       ${(resultados_falla['X'][1] + resultados_falla['Z'][1]) / 0.4:.2f}$       |       ${pd.read_csv(DATA_DIR / 'datos_periodos.csv')['d_t'][0] - resultados_falla['NS'][1]:.2f}$       |        ${resultados_falla['NS'][1]:.2f}$        |
|     2     |   Base    |         ${resultados_base['X'][2]:.2f}$          |        ${resultados_base['Y'][2]:.2f}$         |         ${resultados_base['Z'][2]:.2f}$         |       ${(resultados_base['X'][2] + resultados_base['Z'][2]) / 0.4:.2f}$       |       ${pd.read_csv(DATA_DIR / 'datos_periodos.csv')['d_t'][1] - resultados_base['NS'][2]:.2f}$       |        ${resultados_base['NS'][2]:.2f}$        |
|     2     |   Falla   |         ${resultados_falla['X'][2]:.2f}$          |        ${resultados_falla['Y'][2]:.2f}$         |         ${resultados_falla['Z'][2]:.2f}$         |       ${(resultados_falla['X'][2] + resultados_falla['Z'][2]) / 0.4:.2f}$       |       ${pd.read_csv(DATA_DIR / 'datos_periodos.csv')['d_t'][1] - resultados_falla['NS'][2]:.2f}$       |        ${resultados_falla['NS'][2]:.2f}$        |
|     3     |   Base    |         ${resultados_base['X'][3]:.2f}$          |        ${resultados_base['Y'][3]:.2f}$         |         ${resultados_base['Z'][3]:.2f}$         |       ${(resultados_base['X'][3] + resultados_base['Z'][3]) / 0.4:.2f}$       |       ${pd.read_csv(DATA_DIR / 'datos_periodos.csv')['d_t'][2] - resultados_base['NS'][3]:.2f}$       |        ${resultados_base['NS'][3]:.2f}$        |
|     3     |   Falla   |         ${resultados_falla['X'][3]:.2f}$          |        ${resultados_falla['Y'][3]:.2f}$        |        ${resultados_falla['Z'][3]:.2f}$         |       ${(resultados_falla['X'][3] + resultados_falla['Z'][3]) / 0.4:.2f}$       |       ${pd.read_csv(DATA_DIR / 'datos_periodos.csv')['d_t'][2] - resultados_falla['NS'][3]:.2f}$       |        ${resultados_falla['NS'][3]:.2f}$        |
|     4     |   Base    |         ${resultados_base['X'][4]:.2f}$          |        ${resultados_base['Y'][4]:.2f}$         |         ${resultados_base['Z'][4]:.2f}$         |       ${(resultados_base['X'][4] + resultados_base['Z'][4]) / 0.4:.2f}$       |       ${pd.read_csv(DATA_DIR / 'datos_periodos.csv')['d_t'][3] - resultados_base['NS'][4]:.2f}$       |        ${resultados_base['NS'][4]:.2f}$        |
|     4     |   Falla   |         ${resultados_falla['X'][4]:.2f}$          |        ${resultados_falla['Y'][4]:.2f}$         |         ${resultados_falla['Z'][4]:.2f}$         |       ${(resultados_falla['X'][4] + resultados_falla['Z'][4]) / 0.4:.2f}$       |       ${pd.read_csv(DATA_DIR / 'datos_periodos.csv')['d_t'][3] - resultados_falla['NS'][4]:.2f}$       |        ${resultados_falla['NS'][4]:.2f}$        |
|     5     |   Base    |         ${resultados_base['X'][5]:.2f}$          |        ${resultados_base['Y'][5]:.2f}$         |         ${resultados_base['Z'][5]:.2f}$         |       ${(resultados_base['X'][5] + resultados_base['Z'][5]) / 0.4:.2f}$       |       ${pd.read_csv(DATA_DIR / 'datos_periodos.csv')['d_t'][4] - resultados_base['NS'][5]:.2f}$       |        ${resultados_base['NS'][5]:.2f}$        |
|     5     |   Falla   |         ${resultados_falla['X'][5]:.2f}$          |        ${resultados_falla['Y'][5]:.2f}$         |         ${resultados_falla['Z'][5]:.2f}$         |       ${(resultados_falla['X'][5] + resultados_falla['Z'][5]) / 0.4:.2f}$       |       ${pd.read_csv(DATA_DIR / 'datos_periodos.csv')['d_t'][4] - resultados_falla['NS'][5]:.2f}$       |        ${resultados_falla['NS'][5]:.2f}$        |
| **Total** | **Base**  |       **${sum(resultados_base['X'].values()):.2f}$**       |      **${sum(resultados_base['Y'].values()):.2f}$**      |      **${sum(resultados_base['Z'].values()):.2f}$**       |    **${sum((resultados_base['X'][t] + resultados_base['Z'][t]) / 0.4 for t in range(1, 6)):.2f}$**     |    **${sum(pd.read_csv(DATA_DIR / 'datos_periodos.csv')['d_t'][t-1] - resultados_base['NS'][t] for t in range(1, 6)):.2f}$**     |      **${sum(resultados_base['NS'].values()):.2f}$**      |
| **Total** | **Falla** |       **${sum(resultados_falla['X'].values()):.2f}$**       |      **${sum(resultados_falla['Y'].values()):.2f}$**      |      **${sum(resultados_falla['Z'].values()):.2f}$**       |    **${sum((resultados_falla['X'][t] + resultados_falla['Z'][t]) / 0.4 for t in range(1, 6)):.2f}$**     |    **${sum(pd.read_csv(DATA_DIR / 'datos_periodos.csv')['d_t'][t-1] - resultados_falla['NS'][t] for t in range(1, 6)):.2f}$**     |      **${sum(resultados_falla['NS'].values()):.2f}$**      |

### 2. Inventarios por Periodo

| Periodo | Caso  | Inv. ropa buen estado (kg) | Inv. ropa mal estado (kg) | Inv. género (kg) | Almacenamiento total (kg) | % Capacidad utilizada |
| :-----: | :---: | :------------------------: | :-----------------------: | :--------------: | :-----------------------: | :-------------------: |
|    1    | Base  |            ${resultados_base['IB'][1]:.2f}$            |           ${resultados_base['IM'][1]:.2f}$            |       ${resultados_base['IG'][1]:.2f}$       |           ${resultados_base['IB'][1] + resultados_base['IM'][1] + resultados_base['IG'][1]:.2f}$            |         ${(resultados_base['IB'][1] + resultados_base['IM'][1] + resultados_base['IG'][1]) / 460 * 100:.2f}$          |
|    1    | Falla |            ${resultados_falla['IB'][1]:.2f}$            |           ${resultados_falla['IM'][1]:.2f}$           |       ${resultados_falla['IG'][1]:.2f}$       |           ${resultados_falla['IB'][1] + resultados_falla['IM'][1] + resultados_falla['IG'][1]:.2f}$           |         ${(resultados_falla['IB'][1] + resultados_falla['IM'][1] + resultados_falla['IG'][1]) / 460 * 100:.2f}$          |
|    2    | Base  |            ${resultados_base['IB'][2]:.2f}$            |           ${resultados_base['IM'][2]:.2f}$            |       ${resultados_base['IG'][2]:.2f}$       |           ${resultados_base['IB'][2] + resultados_base['IM'][2] + resultados_base['IG'][2]:.2f}$            |         ${(resultados_base['IB'][2] + resultados_base['IM'][2] + resultados_base['IG'][2]) / 460 * 100:.2f}$          |
|    2    | Falla |            ${resultados_falla['IB'][2]:.2f}$            |           ${resultados_falla['IM'][2]:.2f}$            |       ${resultados_falla['IG'][2]:.2f}$       |           ${resultados_falla['IB'][2] + resultados_falla['IM'][2] + resultados_falla['IG'][2]:.2f}$            |         ${(resultados_falla['IB'][2] + resultados_falla['IM'][2] + resultados_falla['IG'][2]) / 460 * 100:.2f}$          |
|    3    | Base  |            ${resultados_base['IB'][3]:.2f}$            |           ${resultados_base['IM'][3]:.2f}$           |       ${resultados_base['IG'][3]:.2f}$       |           ${resultados_base['IB'][3] + resultados_base['IM'][3] + resultados_base['IG'][3]:.2f}$           |         ${(resultados_base['IB'][3] + resultados_base['IM'][3] + resultados_base['IG'][3]) / 460 * 100:.2f}$          |
|    3    | Falla |            ${resultados_falla['IB'][3]:.2f}$            |           ${resultados_falla['IM'][3]:.2f}$            |       ${resultados_falla['IG'][3]:.2f}$       |           ${resultados_falla['IB'][3] + resultados_falla['IM'][3] + resultados_falla['IG'][3]:.2f}$            |         ${(resultados_falla['IB'][3] + resultados_falla['IM'][3] + resultados_falla['IG'][3]) / 460 * 100:.2f}$          |
|    4    | Base  |            ${resultados_base['IB'][4]:.2f}$            |           ${resultados_base['IM'][4]:.2f}$            |       ${resultados_base['IG'][4]:.2f}$       |           ${resultados_base['IB'][4] + resultados_base['IM'][4] + resultados_base['IG'][4]:.2f}$            |         ${(resultados_base['IB'][4] + resultados_base['IM'][4] + resultados_base['IG'][4]) / 460 * 100:.2f}$          |
|    4    | Falla |            ${resultados_falla['IB'][4]:.2f}$            |           ${resultados_falla['IM'][4]:.2f}$            |       ${resultados_falla['IG'][4]:.2f}$       |           ${resultados_falla['IB'][4] + resultados_falla['IM'][4] + resultados_falla['IG'][4]:.2f}$            |         ${(resultados_falla['IB'][4] + resultados_falla['IM'][4] + resultados_falla['IG'][4]) / 460 * 100:.2f}$          |
|    5    | Base  |            ${resultados_base['IB'][5]:.2f}$            |           ${resultados_base['IM'][5]:.2f}$            |       ${resultados_base['IG'][5]:.2f}$       |           ${resultados_base['IB'][5] + resultados_base['IM'][5] + resultados_base['IG'][5]:.2f}$            |         ${(resultados_base['IB'][5] + resultados_base['IM'][5] + resultados_base['IG'][5]) / 460 * 100:.2f}$          |
|    5    | Falla |            ${resultados_falla['IB'][5]:.2f}$            |           ${resultados_falla['IM'][5]:.2f}$            |       ${resultados_falla['IG'][5]:.2f}$       |           ${resultados_falla['IB'][5] + resultados_falla['IM'][5] + resultados_falla['IG'][5]:.2f}$            |         ${(resultados_falla['IB'][5] + resultados_falla['IM'][5] + resultados_falla['IG'][5]) / 460 * 100:.2f}$          |

### 3. Recursos Humanos y Utilización

|  Periodo  |   Caso    | Trabajadores contratados | Trabajadores por boleta | Total trabajadores | Horas disponibles | Horas utilizadas | % Utilización |
| :-------: | :-------: | :----------------------: | :---------------------: | :----------------: | :---------------: | :--------------: | :-----------: |
|     1     |   Base    |            ${2}$             |            ${resultados_base['W'][1]}$            |         ${2 + resultados_base['W'][1]}$          |       ${8 * (2 + resultados_base['W'][1]):.2f}$       |      ${0.17 * resultados_base['Y'][1] + 0.19 * resultados_base['Z'][1]:.2f}$       |    ${(0.17 * resultados_base['Y'][1] + 0.19 * resultados_base['Z'][1]) / (8 * (2 + resultados_base['W'][1])) * 100:.2f}$     |
|     1     |   Falla   |            ${2}$             |            ${resultados_falla['W'][1]}$            |         ${2 + resultados_falla['W'][1]}$          |       ${8 * (2 + resultados_falla['W'][1]):.2f}$       |      ${0.17 * 1.25 * resultados_falla['Y'][1] + 0.19 * resultados_falla['Z'][1]:.2f}$       |    ${(0.17 * 1.25 * resultados_falla['Y'][1] + 0.19 * resultados_falla['Z'][1]) / (8 * (2 + resultados_falla['W'][1])) * 100:.2f}$     |
|     2     |   Base    |            ${2}$             |            ${resultados_base['W'][2]}$            |         ${2 + resultados_base['W'][2]}$          |       ${8 * (2 + resultados_base['W'][2]):.2f}$       |      ${0.17 * resultados_base['Y'][2] + 0.19 * resultados_base['Z'][2]:.2f}$       |    ${(0.17 * resultados_base['Y'][2] + 0.19 * resultados_base['Z'][2]) / (8 * (2 + resultados_base['W'][2])) * 100:.2f}$     |
|     2     |   Falla   |            ${2}$             |            ${resultados_falla['W'][2]}$            |         ${2 + resultados_falla['W'][2]}$          |       ${8 * (2 + resultados_falla['W'][2]):.2f}$       |      ${0.17 * 1.25 * resultados_falla['Y'][2] + 0.19 * resultados_falla['Z'][2]:.2f}$       |    ${(0.17 * 1.25 * resultados_falla['Y'][2] + 0.19 * resultados_falla['Z'][2]) / (8 * (2 + resultados_falla['W'][2])) * 100:.2f}$     |
|     3     |   Base    |            ${2}$             |            ${resultados_base['W'][3]}$            |         ${2 + resultados_base['W'][3]}$          |       ${8 * (2 + resultados_base['W'][3]):.2f}$       |      ${0.17 * resultados_base['Y'][3] + 0.19 * resultados_base['Z'][3]:.2f}$       |    ${(0.17 * resultados_base['Y'][3] + 0.19 * resultados_base['Z'][3]) / (8 * (2 + resultados_base['W'][3])) * 100:.2f}$     |
|     3     |   Falla   |            ${2}$             |            ${resultados_falla['W'][3]}$            |         ${2 + resultados_falla['W'][3]}$          |       ${8 * (2 + resultados_falla['W'][3]):.2f}$       |      ${0.17 * 1.25 * resultados_falla['Y'][3] + 0.19 * resultados_falla['Z'][3]:.2f}$       |    ${(0.17 * 1.25 * resultados_falla['Y'][3] + 0.19 * resultados_falla['Z'][3]) / (8 * (2 + resultados_falla['W'][3])) * 100:.2f}$     |
|     4     |   Base    |            ${2}$             |            ${resultados_base['W'][4]}$            |         ${2 + resultados_base['W'][4]}$          |       ${8 * (2 + resultados_base['W'][4]):.2f}$       |      ${0.17 * resultados_base['Y'][4] + 0.19 * resultados_base['Z'][4]:.2f}$       |    ${(0.17 * resultados_base['Y'][4] + 0.19 * resultados_base['Z'][4]) / (8 * (2 + resultados_base['W'][4])) * 100:.2f}$     |
|     4     |   Falla   |            ${2}$             |            ${resultados_falla['W'][4]}$            |         ${2 + resultados_falla['W'][4]}$          |       ${8 * (2 + resultados_falla['W'][4]):.2f}$       |      ${0.17 * 1.25 * resultados_falla['Y'][4] + 0.19 * resultados_falla['Z'][4]:.2f}$       |    ${(0.17 * 1.25 * resultados_falla['Y'][4] + 0.19 * resultados_falla['Z'][4]) / (8 * (2 + resultados_falla['W'][4])) * 100:.2f}$     |
|     5     |   Base    |            ${2}$             |            ${resultados_base['W'][5]}$            |         ${2 + resultados_base['W'][5]}$          |       ${8 * (2 + resultados_base['W'][5]):.2f}$       |      ${0.17 * resultados_base['Y'][5] + 0.19 * resultados_base['Z'][5]:.2f}$       |    ${(0.17 * resultados_base['Y'][5] + 0.19 * resultados_base['Z'][5]) / (8 * (2 + resultados_base['W'][5])) * 100:.2f}$     |
|     5     |   Falla   |            ${2}$             |            ${resultados_falla['W'][5]}$            |         ${2 + resultados_falla['W'][5]}$          |       ${8 * (2 + resultados_falla['W'][5]):.2f}$       |      ${0.17 * 1.25 * resultados_falla['Y'][5] + 0.19 * resultados_falla['Z'][5]:.2f}$       |    ${(0.17 * 1.25 * resultados_falla['Y'][5] + 0.19 * resultados_falla['Z'][5]) / (8 * (2 + resultados_falla['W'][5])) * 100:.2f}$     |
| **Total** | **Base**  |          **${2 * 5}$**          |          **${sum(resultados_base['W'].values())}$**          |       **${2 * 5 + sum(resultados_base['W'].values())}$**       |    **${8 * (2 * 5 + sum(resultados_base['W'].values())):.2f}$**     |    **${sum(0.17 * resultados_base['Y'][t] + 0.19 * resultados_base['Z'][t] for t in range(1, 6)):.2f}$**    |  **${sum(0.17 * resultados_base['Y'][t] + 0.19 * resultados_base['Z'][t] for t in range(1, 6)) / (8 * (2 * 5 + sum(resultados_base['W'].values()))) * 100:.2f}$**   |
| **Total** | **Falla** |          **${2 * 5}$**          |          **${sum(resultados_falla['W'].values())}$**          |       **${2 * 5 + sum(resultados_falla['W'].values())}$**       |    **${8 * (2 * 5 + sum(resultados_falla['W'].values())):.2f}$**     |    **${sum(0.17 * 1.25 * resultados_falla['Y'][t] + 0.19 * resultados_falla['Z'][t] for t in range(1, 6)):.2f}$**    |  **${sum(0.17 * 1.25 * resultados_falla['Y'][t] + 0.19 * resultados_falla['Z'][t] for t in range(1, 6)) / (8 * (2 * 5 + sum(resultados_falla['W'].values()))) * 100:.2f}$**   |

### 4. Desglose de Costos

|              Componente               |    Caso Base     |             |    Caso Falla    |             | Variación  |
| :-----------------------------------: | :--------------: | :---------: | :--------------: | :---------: | :--------: |
|                                       |    Valor ($)     | Porcentaje  |    Valor ($)     | Porcentaje  | Porcentual |
|          Personal contratado          |    ${11500 * 8 * 2 * 5:,.2f}$    |   ${11500 * 8 * 2 * 5 / resultados_base['valor_objetivo'] * 100:.2f}\\%$    |    ${11500 * 8 * 2 * 5:,.2f}$    |    ${11500 * 8 * 2 * 5 / resultados_falla['valor_objetivo'] * 100:.2f}\\%$    |   ${0.00}\\%$    |
|          Personal por boleta          |   ${215000 * sum(resultados_base['W'].values()):,.2f}$   |   ${215000 * sum(resultados_base['W'].values()) / resultados_base['valor_objetivo'] * 100:.2f}\\%$    |   ${215000 * sum(resultados_falla['W'].values()):,.2f}$   |   ${215000 * sum(resultados_falla['W'].values()) / resultados_falla['valor_objetivo'] * 100:.2f}\\%$    |  ${(215000 * sum(resultados_falla['W'].values()) - 215000 * sum(resultados_base['W'].values())) / (215000 * sum(resultados_base['W'].values())) * 100:.2f}\\%$   |
|        Transformación a género        |    ${395 * sum(resultados_base['Y'].values()):,.2f}$    |    ${395 * sum(resultados_base['Y'].values()) / resultados_base['valor_objetivo'] * 100:.2f}\\%$    |    ${395 * sum(resultados_falla['Y'].values()):,.2f}$    |    ${395 * sum(resultados_falla['Y'].values()) / resultados_falla['valor_objetivo'] * 100:.2f}\\%$    |   ${(395 * sum(resultados_falla['Y'].values()) - 395 * sum(resultados_base['Y'].values())) / (395 * sum(resultados_base['Y'].values())) * 100:.2f}\\%$   |
|         Producción de prendas         |    ${265 * sum(resultados_base['Z'].values()):,.2f}$     |    ${265 * sum(resultados_base['Z'].values()) / resultados_base['valor_objetivo'] * 100:.2f}\\%$    |    ${265 * sum(resultados_falla['Z'].values()):,.2f}$     |    ${265 * sum(resultados_falla['Z'].values()) / resultados_falla['valor_objetivo'] * 100:.2f}\\%$    |   ${(265 * sum(resultados_falla['Z'].values()) - 265 * sum(resultados_base['Z'].values())) / (265 * sum(resultados_base['Z'].values())) * 100:.2f}\\%$   |
|            Almacenamiento             |    ${405 * sum(resultados_base['IB'][t] + resultados_base['IM'][t] + resultados_base['IG'][t] for t in range(1, 6)):,.2f}$     |    ${405 * sum(resultados_base['IB'][t] + resultados_base['IM'][t] + resultados_base['IG'][t] for t in range(1, 6)) / resultados_base['valor_objetivo'] * 100:.2f}\\%$    |    ${405 * sum(resultados_falla['IB'][t] + resultados_falla['IM'][t] + resultados_falla['IG'][t] for t in range(1, 6)):,.2f}$     |    ${405 * sum(resultados_falla['IB'][t] + resultados_falla['IM'][t] + resultados_falla['IG'][t] for t in range(1, 6)) / resultados_falla['valor_objetivo'] * 100:.2f}\\%$    |  ${(405 * sum(resultados_falla['IB'][t] + resultados_falla['IM'][t] + resultados_falla['IG'][t] for t in range(1, 6)) - 405 * sum(resultados_base['IB'][t] + resultados_base['IM'][t] + resultados_base['IG'][t] for t in range(1, 6))) / (405 * sum(resultados_base['IB'][t] + resultados_base['IM'][t] + resultados_base['IG'][t] for t in range(1, 6))) * 100:.2f}\\%$   |
| Penalización por demanda insatisfecha |   ${7000 * sum(resultados_base['NS'].values()):,.2f}$   |   ${7000 * sum(resultados_base['NS'].values()) / resultados_base['valor_objetivo'] * 100:.2f}\\%$    |   ${7000 * sum(resultados_falla['NS'].values()):,.2f}$   |   ${7000 * sum(resultados_falla['NS'].values()) / resultados_falla['valor_objetivo'] * 100:.2f}\\%$    |   ${(7000 * sum(resultados_falla['NS'].values()) - 7000 * sum(resultados_base['NS'].values())) / (7000 * sum(resultados_base['NS'].values())) * 100:.2f}\\%$   |
|            **Costo total**            | **${resultados_base["valor_objetivo"]:,.2f}$** | **${100}\\%$** | **${resultados_falla["valor_objetivo"]:,.2f}$** | **${100}\\%$** | **${((resultados_falla['valor_objetivo'] - resultados_base['valor_objetivo']) / resultados_base['valor_objetivo']) * 100:.2f}\\%$** |

## Impacto en Indicadores Clave

### 1. Costo Total

- Caso base: ${resultados_base['valor_objetivo']:,.2f}$
- Caso con falla: ${resultados_falla['valor_objetivo']:,.2f}$
- Variación: ${cambio_valor_objetivo:+.2f}\\%$

### 2. Procesamiento de Ropa en Mal Estado
- Caso base: ${total_Y_base:.2f}$ kg
- Caso con falla: ${total_Y_falla:.2f}$ kg
- Variación: ${cambio_procesamiento:+.2f}\\%$

### 3. Demanda Insatisfecha
- Caso base: ${total_NS_base:.2f}$ prendas
- Caso con falla: ${total_NS_falla:.2f}$ prendas
- Variación: ${cambio_demanda_insatisfecha:+.2f}\\%$

### 4. Utilización de Personal
- Caso base: ${total_W_base}$ trabajadores por boleta en total
- Caso con falla: ${total_W_falla}$ trabajadores por boleta en total
- Variación: ${cambio_trabajadores:+.2f}\\%$

## Gráficos Comparativos

### Comparación de Producción vs Demanda
![Comparación de Producción vs Demanda](../src/pregunta3/grafico_comparativo_produccion_demanda.png)

*Figura 1: Comparación de la producción y demanda entre el caso base y el caso con falla técnica.*

### Comparación de Recursos Humanos
![Comparación de Recursos Humanos](../src/pregunta3/grafico_comparativo_recursos_humanos.png)

*Figura 2: Comparación de la utilización de recursos humanos entre el caso base y el caso con falla técnica.*

### Comparación de Costos
![Comparación de Costos](../src/pregunta3/grafico_comparativo_costos.png)

*Figura 3: Comparación de los costos totales y su distribución entre el caso base y el caso con falla técnica.*

### Comparación de Procesamiento de Ropa en Mal Estado
![Comparación de Procesamiento](../src/pregunta3/grafico_comparativo_procesamiento.png)

*Figura 4: Comparación del procesamiento de ropa en mal estado entre el caso base y el caso con falla técnica.*

## Análisis Detallado

### Impacto en la Capacidad de Producción
La falla técnica que aumentó el tiempo de procesamiento en un 25% ha tenido los siguientes efectos en la capacidad productiva del sistema:

1. **Eficiencia del Proceso**: El aumento en el tiempo de procesamiento ha resultado en {
    'una reducción' if total_Y_falla < total_Y_base else 'un aumento'} del ${abs(cambio_procesamiento):.2f}\\%$ en la cantidad de ropa en mal estado procesada.

2. **Satisfacción de la Demanda**: La demanda insatisfecha ha {
    'aumentado' if total_NS_falla > total_NS_base else 'disminuido'} en un ${abs(cambio_demanda_insatisfecha):.2f}\\%$, reflejando {
    'una menor' if total_NS_falla > total_NS_base else 'una mayor'} capacidad para cumplir con los requerimientos.

3. **Costos Operativos**: El costo total de operación se ha {
    'incrementado' if cambio_valor_objetivo > 0 else 'reducido'} en un ${abs(cambio_valor_objetivo):.2f}\\%$, principalmente debido a {
    'mayores costos de personal y penalizaciones' if cambio_valor_objetivo > 0 else 'ajustes en la estrategia de producción'}.

### Adaptación de la Estrategia Operativa

El modelo ha respondido a la falla técnica mediante:

1. **Ajuste de Personal**: {
    'Aumentó' if total_W_falla > total_W_base else 'Redujo'} la contratación de trabajadores por boleta en un ${abs(cambio_trabajadores):.2f}\\%$ para {
    'compensar' if total_W_falla > total_W_base else 'optimizar'} la pérdida de eficiencia.

2. **Gestión de Inventarios**: {
    'Se mantuvo una política de inventarios más conservadora' if sum(resultados_falla['IG'].values()) > sum(resultados_base['IG'].values()) else 
    'Se optimizó el uso de inventarios para mantener la eficiencia operativa'}.

## Conclusiones

1. La falla técnica ha tenido un impacto {
    'significativo' if abs(cambio_valor_objetivo) > 10 else 'moderado'} en el costo total de operación.

2. El sistema ha mostrado {
    'una capacidad limitada' if cambio_demanda_insatisfecha > 0 else 'resiliencia'} para adaptarse a la reducción en la eficiencia del procesamiento.

3. Las principales consecuencias se observan en:
   - {"El aumento" if cambio_valor_objetivo > 0 else "La reducción"} del costo total (${cambio_valor_objetivo:+.2f}\\%$)
   - {"El incremento" if cambio_demanda_insatisfecha > 0 else "La disminución"} de la demanda insatisfecha (${cambio_demanda_insatisfecha:+.2f}\\%$)
   - {"La mayor" if cambio_trabajadores > 0 else "La menor"} necesidad de personal (${cambio_trabajadores:+.2f}\\%$)

## Recomendaciones

1. {
    "Evaluar la posibilidad de reparar la máquina para recuperar la eficiencia original del proceso." if cambio_valor_objetivo > 0 else 
    "Mantener las adaptaciones implementadas que han permitido minimizar el impacto de la falla."
}

2. {
    "Considerar la contratación temporal de personal adicional para compensar la pérdida de eficiencia." if cambio_demanda_insatisfecha > 0 else 
    "Mantener los niveles actuales de personal que han demostrado ser efectivos."
}

3. {
    "Revisar y optimizar los procesos de inventario para reducir costos operativos." if sum(resultados_falla['IG'].values()) > sum(resultados_base['IG'].values()) else 
    "Continuar con la política actual de gestión de inventarios que ha demostrado ser eficiente."
}
"""
    
    # Guardar el informe en un archivo Markdown
    os.makedirs(RESPUESTAS_DIR, exist_ok=True)
    with open(RESPUESTAS_DIR / 'pregunta3_falla.md', 'w') as f:
        f.write(informe)
    
    return informe

if __name__ == "__main__":
    print("Resolviendo el modelo de optimización para la pregunta 3...")
    
    # Asegurarse que exista la carpeta para los resultados
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Resolver ambos modelos
    print("Resolviendo caso base...")
    resultados_base, param_base, T_base, kb_base, km_base, d_base = resolver_modelo_base()
    
    print("Resolviendo caso con falla técnica...")
    resultados_falla, param_falla, T_falla, kb_falla, km_falla, d_falla = resolver_modelo_falla()
    
    if resultados_base is not None and resultados_falla is not None:
        print(f"\nModelos resueltos correctamente.")
        print(f"Valor óptimo caso base: ${resultados_base['valor_objetivo']:,.2f}")
        print(f"Valor óptimo con falla: ${resultados_falla['valor_objetivo']:,.2f}")
        
        # Generar informe comparativo
        print("\nGenerando informe comparativo...")
        informe = generar_informe_comparativo(resultados_base, resultados_falla)
        
        print(f"\nSe han generado los gráficos comparativos en: {OUTPUT_DIR}")
        print(f"Se ha generado el informe en: {RESPUESTAS_DIR}/pregunta3_falla.md")
        
        # Mostrar un resumen comparativo básico
        total_Y_base = sum(resultados_base['Y'].values())
        total_Y_falla = sum(resultados_falla['Y'].values())
        cambio_procesamiento = ((total_Y_falla - total_Y_base) / total_Y_base) * 100 if total_Y_base > 0 else float('inf')
        
        total_NS_base = sum(resultados_base['NS'].values())
        total_NS_falla = sum(resultados_falla['NS'].values())
        cambio_demanda_insatisfecha = ((total_NS_falla - total_NS_base) / total_NS_base) * 100 if total_NS_base > 0 else float('inf')
        
        print("\nResumen comparativo:")
        print(f"- Procesamiento de ropa en mal estado: {total_Y_base:.2f} kg (base) vs {total_Y_falla:.2f} kg (con falla) - Cambio: {cambio_procesamiento:+.2f}%")
        print(f"- Demanda insatisfecha: {total_NS_base:.2f} prendas (base) vs {total_NS_falla:.2f} prendas (con falla) - Cambio: {cambio_demanda_insatisfecha:+.2f}%")
        print(f"- Costo total: ${resultados_base['valor_objetivo']:,.2f} (base) vs ${resultados_falla['valor_objetivo']:,.2f} (con falla) - Cambio: {((resultados_falla['valor_objetivo'] - resultados_base['valor_objetivo']) / resultados_base['valor_objetivo']) * 100:+.2f}%")
    else:
        print("No se pudieron resolver los modelos. Verifique los datos y restricciones.")

