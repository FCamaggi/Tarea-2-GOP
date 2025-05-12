#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Módulo para el análisis de sensibilidad de la demanda (Pregunta 6).
Este script resuelve el modelo de optimización para diferentes escenarios de demanda:
- Escenario base (100% de la demanda pronosticada)
- Demanda reducida (80% de la demanda pronosticada)
- Demanda incrementada (120% de la demanda pronosticada)
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
OUTPUT_DIR = BASE_DIR / "desarrollo" / "src" / "pregunta6"
RESPUESTAS_DIR = BASE_DIR / "desarrollo" / "respuestas"

def resolver_modelo_general(demanda_override=None):
    """
    Resuelve el modelo de optimización con los parámetros dados.
    Si se proporciona tau_g_override, usa ese valor en lugar del valor por defecto.
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

     # Si se proporciona un valor específico para tau_g, usarlo
    if demanda_override is not None:
        for i in range(len(d)):
          d[i] = int(d[i] *  demanda_override)

    # aumentar la demanda por demanda override

    
    
    
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
    """Resuelve el modelo de optimización lineal base sin demanda variable"""
   
    return resolver_modelo_general()

def resolver_modelo_demanda_variable(variabilidad):
    """Resuelve el modelo de optimización considerando la demanda variable"""
    return resolver_modelo_general(variabilidad)

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

def generar_graficos_comparativos(resultados_base, resultados_80, resultados_120):
    """Genera gráficos comparativos entre el caso base, demanda reducida (80%) y demanda incrementada (120%)"""
    # Configuración de estilo para los gráficos
    sns.set_theme(style="darkgrid")
    
    # Definir una paleta de colores personalizada más profesional
    paleta_colores = {
        'azul_principal': '#2c3e50',
        'verde_exito': '#27ae60',
        'rojo_error': '#c0392b',
        'naranja_alerta': '#e67e22',
        'azul_claro': '#3498db',
        'gris_neutro': '#95a5a6',
        'morado': '#8e44ad',
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
            (resultados_base[0]['X'][t] + resultados_base[0]['Z'][t]) / param['p'] 
            for t in periodos
        ],
        'Demanda Satisfecha': [
            pd.read_csv(DATA_DIR / 'datos_periodos.csv')['d_t'][t-1] - resultados_base[0]['NS'][t] 
            for t in periodos
        ],
        'Demanda Insatisfecha': [resultados_base[0]['NS'][t] for t in periodos],
        'Demanda Total': [pd.read_csv(DATA_DIR / 'datos_periodos.csv')['d_t'][t-1] for t in periodos]
    })
    
    df_prod_80 = pd.DataFrame({
        'Periodo': periodos,
        'Prendas Producidas': [
            (resultados_80[0]['X'][t] + resultados_80[0]['Z'][t]) / param['p'] 
            for t in periodos
        ],
        'Demanda Satisfecha': [
            int(pd.read_csv(DATA_DIR / 'datos_periodos.csv')['d_t'][t-1] * 0.8) - resultados_80[0]['NS'][t] 
            for t in periodos
        ],
        'Demanda Insatisfecha': [resultados_80[0]['NS'][t] for t in periodos],
        'Demanda Total': [int(pd.read_csv(DATA_DIR / 'datos_periodos.csv')['d_t'][t-1] * 0.8) for t in periodos]
    })
    
    df_prod_120 = pd.DataFrame({
        'Periodo': periodos,
        'Prendas Producidas': [
            (resultados_120[0]['X'][t] + resultados_120[0]['Z'][t]) / param['p'] 
            for t in periodos
        ],
        'Demanda Satisfecha': [
            int(pd.read_csv(DATA_DIR / 'datos_periodos.csv')['d_t'][t-1] * 1.2) - resultados_120[0]['NS'][t] 
            for t in periodos
        ],
        'Demanda Insatisfecha': [resultados_120[0]['NS'][t] for t in periodos],
        'Demanda Total': [int(pd.read_csv(DATA_DIR / 'datos_periodos.csv')['d_t'][t-1] * 1.2) for t in periodos]
    })
    
    # Para recursos humanos
    df_rec_base = pd.DataFrame({
        'Periodo': periodos,
        'Total Trabajadores': [param['w0'] + resultados_base[0]['W'][t] for t in periodos],
        'Trabajadores por Boleta': [resultados_base[0]['W'][t] for t in periodos],
        '% Utilización': [
            (param['tau_g'] * resultados_base[0]['Y'][t] + param['tau_n'] * resultados_base[0]['Z'][t]) / 
            (param['h'] * (param['w0'] + resultados_base[0]['W'][t])) * 100 if (param['w0'] + resultados_base[0]['W'][t]) > 0 else 0
            for t in periodos
        ]
    })
    
    df_rec_80 = pd.DataFrame({
        'Periodo': periodos,
        'Total Trabajadores': [param['w0'] + resultados_80[0]['W'][t] for t in periodos],
        'Trabajadores por Boleta': [resultados_80[0]['W'][t] for t in periodos],
        '% Utilización': [
            (param['tau_g'] * resultados_80[0]['Y'][t] + param['tau_n'] * resultados_80[0]['Z'][t]) / 
            (param['h'] * (param['w0'] + resultados_80[0]['W'][t])) * 100 if (param['w0'] + resultados_80[0]['W'][t]) > 0 else 0
            for t in periodos
        ]
    })
    
    df_rec_120 = pd.DataFrame({
        'Periodo': periodos,
        'Total Trabajadores': [param['w0'] + resultados_120[0]['W'][t] for t in periodos],
        'Trabajadores por Boleta': [resultados_120[0]['W'][t] for t in periodos],
        '% Utilización': [
            (param['tau_g'] * resultados_120[0]['Y'][t] + param['tau_n'] * resultados_120[0]['Z'][t]) / 
            (param['h'] * (param['w0'] + resultados_120[0]['W'][t])) * 100 if (param['w0'] + resultados_120[0]['W'][t]) > 0 else 0
            for t in periodos
        ]
    })
    
    # Para procesamiento de ropa en mal estado
    df_proc_base = pd.DataFrame({
        'Periodo': periodos,
        'Ropa Mal Estado Procesada (kg)': [resultados_base[0]['Y'][t] for t in periodos]
    })
    
    df_proc_80 = pd.DataFrame({
        'Periodo': periodos,
        'Ropa Mal Estado Procesada (kg)': [resultados_80[0]['Y'][t] for t in periodos]
    })
    
    df_proc_120 = pd.DataFrame({
        'Periodo': periodos,
        'Ropa Mal Estado Procesada (kg)': [resultados_120[0]['Y'][t] for t in periodos]
    })
    
    # =====================================================
    # 1. Gráfico comparativo de Producción vs Demanda
    # =====================================================
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6), facecolor='white')
    fig.suptitle('Comparación de Producción y Demanda para Diferentes Escenarios de Demanda', 
                fontsize=16, fontweight='bold')
    
    # Gráfico para caso base
    ax1.set_facecolor('#f8f9fa')
    ax1.set_title('Caso Base (100% Demanda)', fontsize=14, fontweight='bold')
    
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
    
    # Gráfico para caso con demanda reducida (80%)
    ax2.set_facecolor('#f8f9fa')
    ax2.set_title('Demanda Reducida (80%)', fontsize=14, fontweight='bold')
    
    # Crear gráfico apilado para la demanda (caso 80%)
    ax2.bar(df_prod_80['Periodo'], df_prod_80['Demanda Satisfecha'], 
            label='Demanda Satisfecha', alpha=0.8, color=paleta_colores['verde_exito'])
    ax2.bar(df_prod_80['Periodo'], df_prod_80['Demanda Insatisfecha'], 
            bottom=df_prod_80['Demanda Satisfecha'], label='Demanda Insatisfecha', 
            alpha=0.8, color=paleta_colores['rojo_error'])
    
    # Añadir línea para producción (caso 80%)
    ax2.plot(df_prod_80['Periodo'], df_prod_80['Prendas Producidas'], 'o-', 
            color=paleta_colores['azul_principal'], linewidth=2.5, label='Prendas Producidas', 
            markersize=8, path_effects=[patheffects.withStroke(linewidth=4, foreground='white')])
    
    # Etiquetas y configuración (caso 80%)
    ax2.set_xlabel('Periodo', fontsize=12)
    ax2.set_ylabel('Cantidad de Prendas', fontsize=12)
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # Gráfico para caso con demanda incrementada (120%)
    ax3.set_facecolor('#f8f9fa')
    ax3.set_title('Demanda Incrementada (120%)', fontsize=14, fontweight='bold')
    
    # Crear gráfico apilado para la demanda (caso 120%)
    ax3.bar(df_prod_120['Periodo'], df_prod_120['Demanda Satisfecha'], 
            label='Demanda Satisfecha', alpha=0.8, color=paleta_colores['verde_exito'])
    ax3.bar(df_prod_120['Periodo'], df_prod_120['Demanda Insatisfecha'], 
            bottom=df_prod_120['Demanda Satisfecha'], label='Demanda Insatisfecha', 
            alpha=0.8, color=paleta_colores['rojo_error'])
    
    # Añadir línea para producción (caso 120%)
    ax3.plot(df_prod_120['Periodo'], df_prod_120['Prendas Producidas'], 'o-', 
            color=paleta_colores['azul_principal'], linewidth=2.5, label='Prendas Producidas', 
            markersize=8, path_effects=[patheffects.withStroke(linewidth=4, foreground='white')])
    
    # Etiquetas y configuración (caso 120%)
    ax3.set_xlabel('Periodo', fontsize=12)
    ax3.set_ylabel('Cantidad de Prendas', fontsize=12)
    ax3.legend(loc='upper left', fontsize=10)
    ax3.grid(True, alpha=0.3, linestyle='--')
    
    # Ajustar el diseño y guardar con más espacio
    plt.tight_layout(pad=2.0)
    plt.subplots_adjust(wspace=0.3)  # Más espacio entre los subplots
    plt.savefig(OUTPUT_DIR / 'grafico_comparativo_produccion_demanda.png', dpi=300, bbox_inches='tight')
    
    # =====================================================
    # 2. Gráfico comparativo de Recursos Humanos
    # =====================================================
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6), facecolor='white')
    fig.suptitle('Comparación de Recursos Humanos para Diferentes Escenarios de Demanda', 
                fontsize=16, fontweight='bold')
    
    # Gráfico para caso base
    ax1.set_facecolor('#f8f9fa')
    ax1.set_title('Caso Base (100% Demanda)', fontsize=14, fontweight='bold')
    
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
    ax4 = ax1.twinx()
    ax4.plot(df_rec_base['Periodo'], df_rec_base['% Utilización'], 'o-', 
            color=paleta_colores['verde_exito'], linewidth=2.5, label='% Utilización')
    
    # Configuración de ejes y etiquetas (caso base)
    ax1.set_xlabel('Periodo', fontsize=12)
    ax1.set_ylabel('Número de Trabajadores', fontsize=12)
    ax4.set_ylabel('% Utilización', fontsize=12, color=paleta_colores['verde_exito'])
    ax4.set_ylim(0, 105)  # Ajustar para que 100% sea visible
    
    # Combinar leyendas
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines4, labels4 = ax4.get_legend_handles_labels()
    ax1.legend(lines1 + lines4, labels1 + labels4, loc='upper left', fontsize=10)
    
    # Gráfico para caso con demanda reducida (80%)
    ax2.set_facecolor('#f8f9fa')
    ax2.set_title('Demanda Reducida (80%)', fontsize=14, fontweight='bold')
    
    # Crear gráfico de barras para trabajadores (caso 80%)
    ax2.bar(df_rec_80['Periodo'], [param['w0']] * len(periodos), width,
            label='Trabajadores Contratados', color=paleta_colores['azul_principal'])
    ax2.bar(df_rec_80['Periodo'], df_rec_80['Trabajadores por Boleta'], width,
            bottom=[param['w0']] * len(periodos), label='Trabajadores por Boleta',
            color=paleta_colores['naranja_alerta'])
    
    # Añadir etiquetas con el total de trabajadores (caso 80%)
    for i, periodo in enumerate(periodos):
        ax2.annotate(f'Total: {df_rec_80["Total Trabajadores"].iloc[i]}',
                    xy=(periodo, df_rec_80['Total Trabajadores'].iloc[i] + 0.3),
                    ha='center', va='bottom', fontweight='bold')
    
    # Añadir línea para porcentaje de utilización (caso 80%)
    ax5 = ax2.twinx()
    ax5.plot(df_rec_80['Periodo'], df_rec_80['% Utilización'], 'o-', 
            color=paleta_colores['verde_exito'], linewidth=2.5, label='% Utilización')
    
    # Configuración de ejes y etiquetas (caso 80%)
    ax2.set_xlabel('Periodo', fontsize=12)
    ax2.set_ylabel('Número de Trabajadores', fontsize=12)
    ax5.set_ylabel('% Utilización', fontsize=12, color=paleta_colores['verde_exito'])
    ax5.set_ylim(0, 105)  # Ajustar para que 100% sea visible
    
    # Combinar leyendas
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines5, labels5 = ax5.get_legend_handles_labels()
    ax2.legend(lines2 + lines5, labels2 + labels5, loc='upper left', fontsize=10)
    
    # Gráfico para caso con demanda incrementada (120%)
    ax3.set_facecolor('#f8f9fa')
    ax3.set_title('Demanda Incrementada (120%)', fontsize=14, fontweight='bold')
    
    # Crear gráfico de barras para trabajadores (caso 120%)
    ax3.bar(df_rec_120['Periodo'], [param['w0']] * len(periodos), width,
            label='Trabajadores Contratados', color=paleta_colores['azul_principal'])
    ax3.bar(df_rec_120['Periodo'], df_rec_120['Trabajadores por Boleta'], width,
            bottom=[param['w0']] * len(periodos), label='Trabajadores por Boleta',
            color=paleta_colores['naranja_alerta'])
    
    # Añadir etiquetas con el total de trabajadores (caso 120%)
    for i, periodo in enumerate(periodos):
        ax3.annotate(f'Total: {df_rec_120["Total Trabajadores"].iloc[i]}',
                    xy=(periodo, df_rec_120['Total Trabajadores'].iloc[i] + 0.3),
                    ha='center', va='bottom', fontweight='bold')
    
    # Añadir línea para porcentaje de utilización (caso 120%)
    ax6 = ax3.twinx()
    ax6.plot(df_rec_120['Periodo'], df_rec_120['% Utilización'], 'o-', 
            color=paleta_colores['verde_exito'], linewidth=2.5, label='% Utilización')
    
    # Configuración de ejes y etiquetas (caso 120%)
    ax3.set_xlabel('Periodo', fontsize=12)
    ax3.set_ylabel('Número de Trabajadores', fontsize=12)
    ax6.set_ylabel('% Utilización', fontsize=12, color=paleta_colores['verde_exito'])
    ax6.set_ylim(0, 105)  # Ajustar para que 100% sea visible
    
    # Combinar leyendas
    lines3, labels3 = ax3.get_legend_handles_labels()
    lines6, labels6 = ax6.get_legend_handles_labels()
    ax3.legend(lines3 + lines6, labels3 + labels6, loc='upper left', fontsize=10)
    
    # Ajustar el diseño y guardar con más espacio
    plt.tight_layout(pad=2.0)
    plt.subplots_adjust(wspace=0.3)  # Más espacio entre los subplots
    plt.savefig(OUTPUT_DIR / 'grafico_comparativo_recursos_humanos.png', dpi=300, bbox_inches='tight')
    
    # =====================================================
    # 3. Gráfico comparativo de Costos
    # =====================================================
    # Calcular costos para cada caso
    costos_base = {
        'Personal Fijo': param['cc'] * param['h'] * param['w0'] * 5,
        'Personal por Boleta': param['ct'] * sum(resultados_base[0]['W'].values()),
        'Transformación': param['g'] * sum(resultados_base[0]['Y'].values()),
        'Producción': param['n'] * sum(resultados_base[0]['Z'].values()),
        'Almacenamiento': param['a'] * sum(sum(resultados_base[0][inv].values()) for inv in ['IB', 'IM', 'IG']),
        'Penalización': param['cp'] * sum(resultados_base[0]['NS'].values())
    }
    
    costos_80 = {
        'Personal Fijo': param['cc'] * param['h'] * param['w0'] * 5,
        'Personal por Boleta': param['ct'] * sum(resultados_80[0]['W'].values()),
        'Transformación': param['g'] * sum(resultados_80[0]['Y'].values()),
        'Producción': param['n'] * sum(resultados_80[0]['Z'].values()),
        'Almacenamiento': param['a'] * sum(sum(resultados_80[0][inv].values()) for inv in ['IB', 'IM', 'IG']),
        'Penalización': param['cp'] * sum(resultados_80[0]['NS'].values())
    }
    
    costos_120 = {
        'Personal Fijo': param['cc'] * param['h'] * param['w0'] * 5,
        'Personal por Boleta': param['ct'] * sum(resultados_120[0]['W'].values()),
        'Transformación': param['g'] * sum(resultados_120[0]['Y'].values()),
        'Producción': param['n'] * sum(resultados_120[0]['Z'].values()),
        'Almacenamiento': param['a'] * sum(sum(resultados_120[0][inv].values()) for inv in ['IB', 'IM', 'IG']),
        'Penalización': param['cp'] * sum(resultados_120[0]['NS'].values())
    }
    
    categorias = list(costos_base.keys())
    valores_base = list(costos_base.values())
    valores_80 = list(costos_80.values())
    valores_120 = list(costos_120.values())
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6), facecolor='white')
    fig.suptitle('Comparación de Costos para Diferentes Escenarios de Demanda',
                fontsize=16, fontweight='bold')
    
    # Gráfico de torta para caso base
    ax1.set_facecolor('#f8f9fa')
    ax1.set_title(f'Caso Base - Costo Total: ${resultados_base[0]["valor_objetivo"]:,.2f}', 
                 fontsize=14, fontweight='bold')
    
    # Calcular porcentajes para las etiquetas
    total_base = sum(valores_base)
    porcentajes_base = [100 * v / total_base for v in valores_base]
    
    # Gráfico de torta con etiquetas de porcentaje
    wedges, texts, autotexts = ax1.pie(
        valores_base, 
        labels=None,  # Quitar etiquetas iniciales para colocarlas después
        autopct='%1.1f%%',
        startangle=90,
        colors=plt.cm.tab10.colors[:len(categorias)],
        explode=[0.05 if i == valores_base.index(max(valores_base)) else 0 for i in range(len(valores_base))],
        pctdistance=0.85  # Alejar un poco los porcentajes del centro
    )
    
    # Mejorar legibilidad de las etiquetas de porcentajes
    for autotext in autotexts:
        autotext.set_fontsize(9)
        autotext.set_fontweight('bold')
        autotext.set_color('white')  # Color blanco para mejor visibilidad
        autotext.set_path_effects([patheffects.withStroke(linewidth=2, foreground='black')])  # Contorno negro
    
    # Crear una leyenda para las categorías en lugar de etiquetas directas
    ax1.legend(wedges, categorias, title="Componentes", 
              loc="center left", bbox_to_anchor=(0.85, 0, 0.5, 1))
    
    # Gráfico de torta para caso con demanda reducida (80%)
    ax2.set_facecolor('#f8f9fa')
    ax2.set_title(f'Demanda 80% - Costo Total: ${resultados_80[0]["valor_objetivo"]:,.2f}',
                 fontsize=14, fontweight='bold')
    
    # Calcular porcentajes para las etiquetas
    total_80 = sum(valores_80)
    porcentajes_80 = [100 * v / total_80 for v in valores_80]
    
    # Gráfico de torta con etiquetas de porcentaje
    wedges, texts, autotexts = ax2.pie(
        valores_80, 
        labels=None,  # Quitar etiquetas iniciales para colocarlas después
        autopct='%1.1f%%',
        startangle=90,
        colors=plt.cm.tab10.colors[:len(categorias)],
        explode=[0.05 if i == valores_80.index(max(valores_80)) else 0 for i in range(len(valores_80))],
        pctdistance=0.85  # Alejar un poco los porcentajes del centro
    )
    
    # Mejorar legibilidad de las etiquetas de porcentajes
    for autotext in autotexts:
        autotext.set_fontsize(9)
        autotext.set_fontweight('bold')
        autotext.set_color('white')  # Color blanco para mejor visibilidad
        autotext.set_path_effects([patheffects.withStroke(linewidth=2, foreground='black')])  # Contorno negro
    
    # Crear una leyenda para las categorías en lugar de etiquetas directas
    ax2.legend(wedges, categorias, title="Componentes", 
              loc="center left", bbox_to_anchor=(0.85, 0, 0.5, 1))
    
    # Gráfico de torta para caso con demanda incrementada (120%)
    ax3.set_facecolor('#f8f9fa')
    ax3.set_title(f'Demanda 120% - Costo Total: ${resultados_120[0]["valor_objetivo"]:,.2f}',
                 fontsize=14, fontweight='bold')
    
    # Calcular porcentajes para las etiquetas
    total_120 = sum(valores_120)
    porcentajes_120 = [100 * v / total_120 for v in valores_120]
    
    # Gráfico de torta con etiquetas de porcentaje
    wedges, texts, autotexts = ax3.pie(
        valores_120, 
        labels=None,  # Quitar etiquetas iniciales para colocarlas después
        autopct='%1.1f%%',
        startangle=90,
        colors=plt.cm.tab10.colors[:len(categorias)],
        explode=[0.05 if i == valores_120.index(max(valores_120)) else 0 for i in range(len(valores_120))],
        pctdistance=0.85  # Alejar un poco los porcentajes del centro
    )
    
    # Mejorar legibilidad de las etiquetas de porcentajes
    for autotext in autotexts:
        autotext.set_fontsize(9)
        autotext.set_fontweight('bold')
        autotext.set_color('white')  # Color blanco para mejor visibilidad
        autotext.set_path_effects([patheffects.withStroke(linewidth=2, foreground='black')])  # Contorno negro
    
    # Crear una leyenda para las categorías en lugar de etiquetas directas
    ax3.legend(wedges, categorias, title="Componentes", 
              loc="center left", bbox_to_anchor=(0.85, 0, 0.5, 1))
    
    # Ajustar el diseño, asegurando espacio suficiente para leyendas
    plt.tight_layout(pad=2.0)
    plt.subplots_adjust(wspace=0.3)  # Más espacio entre los subplots
    plt.savefig(OUTPUT_DIR / 'grafico_comparativo_costos.png', dpi=300, bbox_inches='tight')
    
    # =====================================================
    # 4. Gráfico comparativo de Procesamiento de Ropa en Mal Estado
    # =====================================================
    fig, ax = plt.subplots(figsize=(12, 8), facecolor='white')
    ax.set_facecolor('#f8f9fa')
    
    # Configurar ancho de barras y posiciones (hacer las barras un poco más estrechas)
    bar_width = 0.22  # Reducido de 0.25 para dar más espacio entre barras
    index = np.arange(len(periodos))
    
    # Crear barras para cada escenario
    bars1 = ax.bar(index - bar_width, df_proc_base['Ropa Mal Estado Procesada (kg)'],
                  bar_width, label='Caso Base (100%)', color=paleta_colores['azul_principal'])
    bars2 = ax.bar(index, df_proc_80['Ropa Mal Estado Procesada (kg)'],
                  bar_width, label='Demanda 80%', color=paleta_colores['verde_exito'])
    bars3 = ax.bar(index + bar_width, df_proc_120['Ropa Mal Estado Procesada (kg)'],
                  bar_width, label='Demanda 120%', color=paleta_colores['rojo_error'])
    
    # Añadir etiquetas a las barras alternando su posición vertical para evitar superposición
    for i, bars in enumerate([bars1, bars2, bars3]):
        for bar in bars:
            height = bar.get_height()
            vert_offset = 3 + (i * 10)  # Offset vertical variable según grupo de barras
            ax.annotate(f'{height:.1f}',
                       xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, vert_offset),
                       textcoords="offset points",
                       ha='center', va='bottom',
                       fontsize=8, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
    
    # Configuración del gráfico
    ax.set_xlabel('Periodo', fontsize=12, fontweight='bold')
    ax.set_ylabel('Ropa en Mal Estado Procesada (kg)', fontsize=12, fontweight='bold')
    ax.set_title('Comparación del Procesamiento de Ropa en Mal Estado para Diferentes Escenarios de Demanda',
               fontsize=14, fontweight='bold')
    ax.set_xticks(index)
    ax.set_xticklabels([f'Periodo {i}' for i in periodos])
    ax.legend(loc='upper left')
    
    # Añadir una línea horizontal para el total en cada caso
    ax.axhline(y=sum(df_proc_base['Ropa Mal Estado Procesada (kg)']/len(periodos)),
              color=paleta_colores['azul_principal'], linestyle='--', alpha=0.7,
              label=f'Prom. Base: {sum(df_proc_base["Ropa Mal Estado Procesada (kg)"]/len(periodos)):.1f} kg')
    ax.axhline(y=sum(df_proc_80['Ropa Mal Estado Procesada (kg)']/len(periodos)),
              color=paleta_colores['verde_exito'], linestyle='--', alpha=0.7,
              label=f'Prom. 80%: {sum(df_proc_80["Ropa Mal Estado Procesada (kg)"]/len(periodos)):.1f} kg')
    ax.axhline(y=sum(df_proc_120['Ropa Mal Estado Procesada (kg)']/len(periodos)),
              color=paleta_colores['rojo_error'], linestyle='--', alpha=0.7,
              label=f'Prom. 120%: {sum(df_proc_120["Ropa Mal Estado Procesada (kg)"]/len(periodos)):.1f} kg')
    
    # Actualizar leyenda con las líneas promedio
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels, loc='upper left')
    
    # Añadir cuadro de texto con las diferencias porcentuales
    cambio_porcentual_80 = ((sum(df_proc_80['Ropa Mal Estado Procesada (kg)']) - 
                         sum(df_proc_base['Ropa Mal Estado Procesada (kg)'])) / 
                        sum(df_proc_base['Ropa Mal Estado Procesada (kg)'])) * 100
    
    cambio_porcentual_120 = ((sum(df_proc_120['Ropa Mal Estado Procesada (kg)']) - 
                         sum(df_proc_base['Ropa Mal Estado Procesada (kg)'])) / 
                        sum(df_proc_base['Ropa Mal Estado Procesada (kg)'])) * 100
    
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    ax.text(0.02, 0.97, 
           f'Variación con 80%: {cambio_porcentual_80:+.2f}%\nVariación con 120%: {cambio_porcentual_120:+.2f}%',
           transform=ax.transAxes, fontsize=12,
           verticalalignment='top', bbox=props)
    
    # Ajustar el diseño y guardar con más espacio
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout(pad=2.0)
    plt.savefig(OUTPUT_DIR / 'grafico_comparativo_procesamiento.png', dpi=300, bbox_inches='tight')
    
    # =====================================================
    # 5. Gráfico comparativo de Satisfacción de Demanda
    # =====================================================
    fig, ax = plt.subplots(figsize=(12, 6), facecolor='white')
    ax.set_facecolor('#f8f9fa')
    
    # Calcular porcentajes de satisfacción para cada caso
    satisfaccion_base = [(df_prod_base['Demanda Satisfecha'][i] / df_prod_base['Demanda Total'][i]) * 100 
                         if df_prod_base['Demanda Total'][i] > 0 else 0 for i in range(len(periodos))]
    
    satisfaccion_80 = [(df_prod_80['Demanda Satisfecha'][i] / df_prod_80['Demanda Total'][i]) * 100 
                       if df_prod_80['Demanda Total'][i] > 0 else 0 for i in range(len(periodos))]
    
    satisfaccion_120 = [(df_prod_120['Demanda Satisfecha'][i] / df_prod_120['Demanda Total'][i]) * 100 
                        if df_prod_120['Demanda Total'][i] > 0 else 0 for i in range(len(periodos))]
    
    # Crear el gráfico de líneas para cada escenario
    ax.plot(periodos, satisfaccion_base, 'o-', linewidth=3, markersize=10, 
            color=paleta_colores['azul_principal'], label='Caso Base (100%)')
    ax.plot(periodos, satisfaccion_80, 's-', linewidth=3, markersize=10,
            color=paleta_colores['verde_exito'], label='Demanda 80%')
    ax.plot(periodos, satisfaccion_120, 'D-', linewidth=3, markersize=10,
            color=paleta_colores['rojo_error'], label='Demanda 120%')
    
    # Añadir etiquetas para cada punto con offsets ajustados para evitar superposiciones
    for i, periodo in enumerate(periodos):
        # Para caso base, etiqueta arriba
        ax.annotate(f'{satisfaccion_base[i]:.1f}%', 
                   xy=(periodo, satisfaccion_base[i]), 
                   xytext=(0, 10),
                   textcoords='offset points',
                   ha='center', va='bottom',
                   fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
        
        # Para caso 80%, etiqueta a la izquierda
        ax.annotate(f'{satisfaccion_80[i]:.1f}%', 
                   xy=(periodo, satisfaccion_80[i]), 
                   xytext=(-15, 0),
                   textcoords='offset points',
                   ha='right', va='center',
                   fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
        
        # Para caso 120%, etiqueta a la derecha
        ax.annotate(f'{satisfaccion_120[i]:.1f}%', 
                   xy=(periodo, satisfaccion_120[i]), 
                   xytext=(15, 0),
                   textcoords='offset points',
                   ha='left', va='center',
                   fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
    
    # Configuración del gráfico
    ax.set_xlabel('Periodo', fontsize=12, fontweight='bold')
    ax.set_ylabel('Porcentaje de Satisfacción de Demanda', fontsize=12, fontweight='bold')
    ax.set_title('Comparación de Satisfacción de Demanda para Diferentes Escenarios',
               fontsize=14, fontweight='bold')
    ax.set_xticks(periodos)
    ax.set_xticklabels([f'Periodo {i}' for i in periodos])
    ax.set_ylim(0, 110)  # Para que el 100% sea visible y haya espacio para las etiquetas
    ax.axhline(y=100, color='gray', linestyle='--', alpha=0.7)
    ax.legend(loc='lower right')
    
    # Añadir cuadro de texto con los promedios
    promedio_base = sum(satisfaccion_base) / len(satisfaccion_base)
    promedio_80 = sum(satisfaccion_80) / len(satisfaccion_80)
    promedio_120 = sum(satisfaccion_120) / len(satisfaccion_120)
    
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    ax.text(0.02, 0.97, 
           f'Prom. Base: {promedio_base:.1f}%\nProm. 80%: {promedio_80:.1f}%\nProm. 120%: {promedio_120:.1f}%',
           transform=ax.transAxes, fontsize=12,
           verticalalignment='top', bbox=props)
    
    # Ajustar el diseño y guardar
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout(pad=2.0)
    plt.savefig(OUTPUT_DIR / 'grafico_comparativo_satisfaccion_demanda.png', dpi=300, bbox_inches='tight')
    
    print("Gráficos comparativos generados con éxito.")
    
    # Crear carpeta para gráficos si no existe
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def generar_informe_consolidado(resultados_base, resultados_demanda_80, resultados_demanda_120):
    """Genera un informe consolidado comparando los tres escenarios de demanda"""
    # Cargar parámetros del modelo para los cálculos
    parametros = pd.read_csv(DATA_DIR / 'datos_parametros.csv')
    param = dict(zip(parametros['Parametro'], parametros['Valor']))
    
    # Cargar datos de periodos
    periodos_data = pd.read_csv(DATA_DIR / 'datos_periodos.csv')
    
    # Calcular indicadores para comparación
    # Escenario 80%
    total_Y_base = sum(resultados_base[0]['Y'].values())
    total_Y_80 = sum(resultados_demanda_80[0]['Y'].values())
    cambio_procesamiento_80 = ((total_Y_80 - total_Y_base) / total_Y_base) * 100 if total_Y_base > 0 else float('inf')
    
    total_NS_base = sum(resultados_base[0]['NS'].values())
    total_NS_80 = sum(resultados_demanda_80[0]['NS'].values())
    cambio_demanda_insatisfecha_80 = ((total_NS_80 - total_NS_base) / total_NS_base) * 100 if total_NS_base > 0 else float('inf')
    
    total_W_base = sum(resultados_base[0]['W'].values())
    total_W_80 = sum(resultados_demanda_80[0]['W'].values())
    cambio_trabajadores_80 = ((total_W_80 - total_W_base) / total_W_base) * 100 if total_W_base > 0 else float('inf')
    
    cambio_valor_objetivo_80 = ((resultados_demanda_80[0]['valor_objetivo'] - resultados_base[0]['valor_objetivo']) / 
                            resultados_base[0]['valor_objetivo']) * 100
    
    # Escenario 120%
    total_Y_120 = sum(resultados_demanda_120[0]['Y'].values())
    cambio_procesamiento_120 = ((total_Y_120 - total_Y_base) / total_Y_base) * 100 if total_Y_base > 0 else float('inf')
    
    total_NS_120 = sum(resultados_demanda_120[0]['NS'].values())
    cambio_demanda_insatisfecha_120 = ((total_NS_120 - total_NS_base) / total_NS_base) * 100 if total_NS_base > 0 else float('inf')
    
    total_W_120 = sum(resultados_demanda_120[0]['W'].values())
    cambio_trabajadores_120 = ((total_W_120 - total_W_base) / total_W_base) * 100 if total_W_base > 0 else float('inf')
    
    cambio_valor_objetivo_120 = ((resultados_demanda_120[0]['valor_objetivo'] - resultados_base[0]['valor_objetivo']) / 
                                resultados_base[0]['valor_objetivo']) * 100
    
    # Datos de demanda por periodo
    demanda_original = periodos_data['d_t'].tolist()
    demanda_80 = [int(d * 0.8) for d in demanda_original]
    demanda_120 = [int(d * 1.2) for d in demanda_original]
    
    # Valores totales para las tablas
    total_X_base = sum(resultados_base[0]['X'].values())
    total_X_80 = sum(resultados_demanda_80[0]['X'].values())
    total_X_120 = sum(resultados_demanda_120[0]['X'].values())
    
    total_Z_base = sum(resultados_base[0]['Z'].values())
    total_Z_80 = sum(resultados_demanda_80[0]['Z'].values())
    total_Z_120 = sum(resultados_demanda_120[0]['Z'].values())
    
    # Calcular demanda satisfecha total
    total_demanda_base = sum(demanda_original)
    total_demanda_80 = sum(demanda_80)
    total_demanda_120 = sum(demanda_120)
    
    satisfecha_base = total_demanda_base - total_NS_base
    satisfecha_80 = total_demanda_80 - total_NS_80
    satisfecha_120 = total_demanda_120 - total_NS_120
    
    porcentaje_satisfecho_base = (satisfecha_base / total_demanda_base) * 100 if total_demanda_base > 0 else 0
    porcentaje_satisfecho_80 = (satisfecha_80 / total_demanda_80) * 100 if total_demanda_80 > 0 else 0
    porcentaje_satisfecho_120 = (satisfecha_120 / total_demanda_120) * 100 if total_demanda_120 > 0 else 0
    
    # Calcular costos desglosados
    costo_personal_fijo = param['cc'] * param['h'] * param['w0'] * 5  # 5 periodos
    
    costo_boleta_base = param['ct'] * total_W_base
    costo_boleta_80 = param['ct'] * total_W_80
    costo_boleta_120 = param['ct'] * total_W_120
    
    costo_transform_base = param['g'] * total_Y_base
    costo_transform_80 = param['g'] * total_Y_80
    costo_transform_120 = param['g'] * total_Y_120
    
    costo_prod_base = param['n'] * total_Z_base
    costo_prod_80 = param['n'] * total_Z_80
    costo_prod_120 = param['n'] * total_Z_120
    
    # Cálculos de inventario
    inv_total_base = sum(resultados_base[0]['IB'][t] + resultados_base[0]['IM'][t] + resultados_base[0]['IG'][t] for t in range(1, 6))
    inv_total_80 = sum(resultados_demanda_80[0]['IB'][t] + resultados_demanda_80[0]['IM'][t] + resultados_demanda_80[0]['IG'][t] for t in range(1, 6))
    inv_total_120 = sum(resultados_demanda_120[0]['IB'][t] + resultados_demanda_120[0]['IM'][t] + resultados_demanda_120[0]['IG'][t] for t in range(1, 6))
    
    costo_alm_base = param['a'] * inv_total_base
    costo_alm_80 = param['a'] * inv_total_80
    costo_alm_120 = param['a'] * inv_total_120
    
    costo_penal_base = param['cp'] * total_NS_base
    costo_penal_80 = param['cp'] * total_NS_80
    costo_penal_120 = param['cp'] * total_NS_120
    
    # Calcular utilización promedio de capacidad de almacenamiento
    util_cap_base = [(resultados_base[0]['IB'][t] + resultados_base[0]['IM'][t] + resultados_base[0]['IG'][t]) / param['s'] * 100 for t in range(1, 6)]
    util_cap_80 = [(resultados_demanda_80[0]['IB'][t] + resultados_demanda_80[0]['IM'][t] + resultados_demanda_80[0]['IG'][t]) / param['s'] * 100 for t in range(1, 6)]
    util_cap_120 = [(resultados_demanda_120[0]['IB'][t] + resultados_demanda_120[0]['IM'][t] + resultados_demanda_120[0]['IG'][t]) / param['s'] * 100 for t in range(1, 6)]
    
    util_cap_prom_base = sum(util_cap_base) / len(util_cap_base)
    util_cap_prom_80 = sum(util_cap_80) / len(util_cap_80)
    util_cap_prom_120 = sum(util_cap_120) / len(util_cap_120)
    
    # Calcular utilización promedio de recursos humanos
    util_rh_base = [(param['tau_g'] * resultados_base[0]['Y'][t] + param['tau_n'] * resultados_base[0]['Z'][t]) / 
                   (param['h'] * (param['w0'] + resultados_base[0]['W'][t])) * 100 for t in range(1, 6)]
    util_rh_80 = [(param['tau_g'] * resultados_demanda_80[0]['Y'][t] + param['tau_n'] * resultados_demanda_80[0]['Z'][t]) / 
                 (param['h'] * (param['w0'] + resultados_demanda_80[0]['W'][t])) * 100 for t in range(1, 6)]
    util_rh_120 = [(param['tau_g'] * resultados_demanda_120[0]['Y'][t] + param['tau_n'] * resultados_demanda_120[0]['Z'][t]) / 
                  (param['h'] * (param['w0'] + resultados_demanda_120[0]['W'][t])) * 100 for t in range(1, 6)]
    
    util_rh_prom_base = sum(util_rh_base) / len(util_rh_base)
    util_rh_prom_80 = sum(util_rh_80) / len(util_rh_80)
    util_rh_prom_120 = sum(util_rh_120) / len(util_rh_120)
    
    # Costo por prenda satisfecha
    costo_prenda_base = resultados_base[0]['valor_objetivo'] / satisfecha_base if satisfecha_base > 0 else float('inf')
    costo_prenda_80 = resultados_demanda_80[0]['valor_objetivo'] / satisfecha_80 if satisfecha_80 > 0 else float('inf')
    costo_prenda_120 = resultados_demanda_120[0]['valor_objetivo'] / satisfecha_120 if satisfecha_120 > 0 else float('inf')
    
    cambio_costo_prenda_80 = ((costo_prenda_80 - costo_prenda_base) / costo_prenda_base) * 100 if costo_prenda_base > 0 else float('inf')
    cambio_costo_prenda_120 = ((costo_prenda_120 - costo_prenda_base) / costo_prenda_base) * 100 if costo_prenda_base > 0 else float('inf')
    
    # Crear el informe consolidado
    informe = """# Pregunta 6: Análisis de Sensibilidad en la Demanda

## Planteamiento de la pregunta

> ¿Qué sucede si la demanda varía con respecto a lo pronosticado? Evalúe el impacto en la planificación si la demanda es un 80% y un 120% del valor original. Tabule y comente los resultados obtenidos. Redondee la demanda diaria a un número entero.

## Resolución del modelo con variaciones en la demanda

Para este análisis de sensibilidad, se evaluaron tres escenarios de demanda:
1. **Escenario base**: Con la demanda original pronosticada (100%)
2. **Demanda reducida**: Con el 80% de la demanda original
3. **Demanda incrementada**: Con el 120% de la demanda original

A continuación se presentan los resultados obtenidos al resolver el modelo para cada uno de estos escenarios.

### Tabla 1: Comparación de demanda por periodo

| Periodo | Demanda original | Demanda reducida (80%) | Demanda incrementada (120%) |
|---------|------------------|------------------------|---------------------------|
"""
    
    # Agregar filas para cada periodo
    for i, (d_orig, d_80, d_120) in enumerate(zip(demanda_original, demanda_80, demanda_120)):
        informe += f"| {i+1} | {d_orig} | {d_80} | {d_120} |\n"
    
    # Agregar fila de totales
    informe += f"| **Total** | **{total_demanda_base}** | **{total_demanda_80}** | **{total_demanda_120}** |\n\n"
    
    # Agregar tabla de resultados operativos
    informe += """### Tabla 2: Comparación de resultados operativos

| Indicador | Demanda original | Demanda reducida (80%) | Variación (%) | Demanda incrementada (120%) | Variación (%) |
|-----------|------------------|------------------------|---------------|---------------------------|---------------|
| **Procesamiento y producción** | | | | | |
"""

    # Agregar datos de procesamiento y producción
    informe += f"| Ropa en buen estado utilizada (kg) | {total_X_base:.2f} | {total_X_80:.2f} | {((total_X_80 - total_X_base) / total_X_base) * 100 if total_X_base > 0 else 0:.2f}% | {total_X_120:.2f} | {((total_X_120 - total_X_base) / total_X_base) * 100 if total_X_base > 0 else 0:.2f}% |\n"
    informe += f"| Ropa en mal estado procesada (kg) | {total_Y_base:.2f} | {total_Y_80:.2f} | {cambio_procesamiento_80:.2f}% | {total_Y_120:.2f} | {cambio_procesamiento_120:.2f}% |\n"
    informe += f"| Género utilizado (kg) | {total_Z_base:.2f} | {total_Z_80:.2f} | {((total_Z_80 - total_Z_base) / total_Z_base) * 100 if total_Z_base > 0 else 0:.2f}% | {total_Z_120:.2f} | {((total_Z_120 - total_Z_base) / total_Z_base) * 100 if total_Z_base > 0 else 0:.2f}% |\n"
    
    # Recursos humanos
    informe += "| **Recursos humanos** | | | | | |\n"
    informe += f"| Trabajadores por boleta (total) | {total_W_base} | {total_W_80} | {cambio_trabajadores_80:.2f}% | {total_W_120} | {cambio_trabajadores_120:.2f}% |\n"
    informe += f"| Utilización de horas-hombre (promedio) | {util_rh_prom_base:.2f}% | {util_rh_prom_80:.2f}% | {util_rh_prom_80 - util_rh_prom_base:.2f} pts | {util_rh_prom_120:.2f}% | {util_rh_prom_120 - util_rh_prom_base:.2f} pts |\n"
    
    # Demanda
    informe += "| **Demanda** | | | | | |\n"
    informe += f"| Demanda total (prendas) | {total_demanda_base} | {total_demanda_80} | -20.00% | {total_demanda_120} | +20.00% |\n"
    informe += f"| Demanda satisfecha (prendas) | {satisfecha_base:.2f} | {satisfecha_80:.2f} | {((satisfecha_80 - satisfecha_base) / satisfecha_base) * 100 if satisfecha_base > 0 else 0:.2f}% | {satisfecha_120:.2f} | {((satisfecha_120 - satisfecha_base) / satisfecha_base) * 100 if satisfecha_base > 0 else 0:.2f}% |\n"
    informe += f"| Demanda no satisfecha (prendas) | {total_NS_base:.2f} | {total_NS_80:.2f} | {cambio_demanda_insatisfecha_80:.2f}% | {total_NS_120:.2f} | {cambio_demanda_insatisfecha_120:.2f}% |\n"
    informe += f"| % Satisfacción global | {porcentaje_satisfecho_base:.2f}% | {porcentaje_satisfecho_80:.2f}% | {porcentaje_satisfecho_80 - porcentaje_satisfecho_base:.2f} pts | {porcentaje_satisfecho_120:.2f}% | {porcentaje_satisfecho_120 - porcentaje_satisfecho_base:.2f} pts |\n"
    
    # Inventarios
    informe += "| **Inventarios** | | | | | |\n"
    informe += f"| Inventario promedio total (kg) | {inv_total_base/5:.2f} | {inv_total_80/5:.2f} | {((inv_total_80 - inv_total_base) / inv_total_base) * 100 if inv_total_base > 0 else 0:.2f}% | {inv_total_120/5:.2f} | {((inv_total_120 - inv_total_base) / inv_total_base) * 100 if inv_total_base > 0 else 0:.2f}% |\n"
    informe += f"| Utilización promedio capacidad almacenamiento | {util_cap_prom_base:.2f}% | {util_cap_prom_80:.2f}% | {util_cap_prom_80 - util_cap_prom_base:.2f} pts | {util_cap_prom_120:.2f}% | {util_cap_prom_120 - util_cap_prom_base:.2f} pts |\n\n"
    
    # Tabla de costos
    informe += """### Tabla 3: Comparación de costos

| Componente de costo | Demanda original ($) | Demanda reducida (80%) ($) | Variación (%) | Demanda incrementada (120%) ($) | Variación (%) |
|---------------------|---------------------|--------------------------|--------------|--------------------------------|--------------|
"""
    informe += f"| Personal contratado | {costo_personal_fijo:,.2f} | {costo_personal_fijo:,.2f} | 0.00% | {costo_personal_fijo:,.2f} | 0.00% |\n"
    informe += f"| Personal por boleta | {costo_boleta_base:,.2f} | {costo_boleta_80:,.2f} | {((costo_boleta_80 - costo_boleta_base) / costo_boleta_base) * 100 if costo_boleta_base > 0 else 0:.2f}% | {costo_boleta_120:,.2f} | {((costo_boleta_120 - costo_boleta_base) / costo_boleta_base) * 100 if costo_boleta_base > 0 else 0:.2f}% |\n"
    informe += f"| Transformación a género | {costo_transform_base:,.2f} | {costo_transform_80:,.2f} | {((costo_transform_80 - costo_transform_base) / costo_transform_base) * 100 if costo_transform_base > 0 else 0:.2f}% | {costo_transform_120:,.2f} | {((costo_transform_120 - costo_transform_base) / costo_transform_base) * 100 if costo_transform_base > 0 else 0:.2f}% |\n"
    informe += f"| Producción de prendas | {costo_prod_base:,.2f} | {costo_prod_80:,.2f} | {((costo_prod_80 - costo_prod_base) / costo_prod_base) * 100 if costo_prod_base > 0 else 0:.2f}% | {costo_prod_120:,.2f} | {((costo_prod_120 - costo_prod_base) / costo_prod_base) * 100 if costo_prod_base > 0 else 0:.2f}% |\n"
    informe += f"| Almacenamiento | {costo_alm_base:,.2f} | {costo_alm_80:,.2f} | {((costo_alm_80 - costo_alm_base) / costo_alm_base) * 100 if costo_alm_base > 0 else 0:.2f}% | {costo_alm_120:,.2f} | {((costo_alm_120 - costo_alm_base) / costo_alm_base) * 100 if costo_alm_base > 0 else 0:.2f}% |\n"
    informe += f"| Penalización por demanda insatisfecha | {costo_penal_base:,.2f} | {costo_penal_80:,.2f} | {((costo_penal_80 - costo_penal_base) / costo_penal_base) * 100 if costo_penal_base > 0 else 0:.2f}% | {costo_penal_120:,.2f} | {((costo_penal_120 - costo_penal_base) / costo_penal_base) * 100 if costo_penal_base > 0 else 0:.2f}% |\n"
    informe += f"| **Costo total** | **{resultados_base[0]['valor_objetivo']:,.2f}** | **{resultados_demanda_80[0]['valor_objetivo']:,.2f}** | **{cambio_valor_objetivo_80:.2f}%** | **{resultados_demanda_120[0]['valor_objetivo']:,.2f}** | **{cambio_valor_objetivo_120:.2f}%** |\n"
    informe += f"| **Costo por prenda satisfecha** | **{costo_prenda_base:.2f}** | **{costo_prenda_80:.2f}** | **{cambio_costo_prenda_80:.2f}%** | **{costo_prenda_120:.2f}** | **{cambio_costo_prenda_120:.2f}%** |\n\n"
    
    # Agregar gráficos
    informe += """## Gráficos comparativos

### Comparación de Producción vs Demanda
![Comparación de Producción vs Demanda](../src/pregunta6/grafico_comparativo_produccion_demanda.png)

*Figura 1: Comparación de la producción y demanda entre los tres escenarios.*

### Comparación de Recursos Humanos
![Comparación de Recursos Humanos](../src/pregunta6/grafico_comparativo_recursos_humanos.png)

*Figura 2: Comparación de la utilización de recursos humanos entre los tres escenarios.*

### Comparación de Costos
![Comparación de Costos](../src/pregunta6/grafico_comparativo_costos.png)

*Figura 3: Comparación de los costos totales y su distribución entre los tres escenarios.*

### Comparación de Procesamiento de Ropa en Mal Estado
![Comparación de Procesamiento](../src/pregunta6/grafico_comparativo_procesamiento.png)

*Figura 4: Comparación del procesamiento de ropa en mal estado entre los tres escenarios.*

### Comparación de Satisfacción de Demanda
![Comparación de Satisfacción](../src/pregunta6/grafico_comparativo_satisfaccion_demanda.png)

*Figura 5: Comparación del porcentaje de satisfacción de demanda entre los tres escenarios.*

"""

    # Análisis de impacto
    informe += """## Análisis de impacto

### 1. Impacto de una demanda reducida (80%)

La reducción de la demanda en un 20% ha resultado en:

"""

    if cambio_valor_objetivo_80 < 0:
        informe += f"- **Reducción en el costo total**: El costo total se redujo en {abs(cambio_valor_objetivo_80):.2f}%, pasando de ${resultados_base[0]['valor_objetivo']:,.2f} a ${resultados_demanda_80[0]['valor_objetivo']:,.2f}.\n"
    else:
        informe += f"- **Aumento en el costo total**: A pesar de la reducción en la demanda, el costo total aumentó en {cambio_valor_objetivo_80:.2f}%, pasando de ${resultados_base[0]['valor_objetivo']:,.2f} a ${resultados_demanda_80[0]['valor_objetivo']:,.2f}.\n"
        
    if cambio_procesamiento_80 < 0:
        informe += f"- **Reducción en el procesamiento**: La cantidad de ropa en mal estado procesada se redujo en {abs(cambio_procesamiento_80):.2f}%.\n"
    else:
        informe += f"- **Aumento en el procesamiento**: A pesar de la reducción en la demanda, la cantidad de ropa en mal estado procesada aumentó en {cambio_procesamiento_80:.2f}%.\n"
    
    if cambio_trabajadores_80 < 0:
        informe += f"- **Disminución en la contratación de personal**: El número total de trabajadores por boleta disminuyó en {abs(cambio_trabajadores_80):.2f}%.\n"
    else:
        informe += f"- **Aumento en la contratación de personal**: A pesar de la reducción en la demanda, el número total de trabajadores por boleta aumentó en {cambio_trabajadores_80:.2f}%.\n"
        
    if cambio_demanda_insatisfecha_80 < 0:
        informe += f"- **Mejora en la satisfacción de demanda**: La demanda insatisfecha se redujo en {abs(cambio_demanda_insatisfecha_80):.2f}%, con un porcentaje de satisfacción global del {porcentaje_satisfecho_80:.2f}% (un cambio de {porcentaje_satisfecho_80 - porcentaje_satisfecho_base:.2f} puntos porcentuales respecto al escenario base).\n"
    else:
        informe += f"- **Deterioro en la satisfacción de demanda**: La demanda insatisfecha aumentó en {cambio_demanda_insatisfecha_80:.2f}%, con un porcentaje de satisfacción global del {porcentaje_satisfecho_80:.2f}% (un cambio de {porcentaje_satisfecho_80 - porcentaje_satisfecho_base:.2f} puntos porcentuales respecto al escenario base).\n"

    if cambio_costo_prenda_80 < 0:
        informe += f"- **Reducción en el costo por prenda**: El costo por prenda satisfecha se redujo en {abs(cambio_costo_prenda_80):.2f}%, lo que indica la presencia de economías de escala inversa.\n\n"
    else:
        informe += f"- **Aumento en el costo por prenda**: El costo por prenda satisfecha aumentó en {cambio_costo_prenda_80:.2f}%, lo que indica la presencia de economías de escala en el sistema.\n\n"
    
    # Impacto de una demanda incrementada (120%)
    informe += """### 2. Impacto de una demanda incrementada (120%)

El incremento de la demanda en un 20% ha resultado en:

"""

    if cambio_valor_objetivo_120 < 0:
        informe += f"- **Reducción en el costo total**: A pesar del aumento en la demanda, el costo total se redujo en {abs(cambio_valor_objetivo_120):.2f}%, pasando de ${resultados_base[0]['valor_objetivo']:,.2f} a ${resultados_demanda_120[0]['valor_objetivo']:,.2f}.\n"
    else:
        informe += f"- **Aumento en el costo total**: El costo total aumentó en {cambio_valor_objetivo_120:.2f}%, pasando de ${resultados_base[0]['valor_objetivo']:,.2f} a ${resultados_demanda_120[0]['valor_objetivo']:,.2f}.\n"
        
    if cambio_procesamiento_120 < 0:
        informe += f"- **Reducción en el procesamiento**: A pesar del aumento en la demanda, la cantidad de ropa en mal estado procesada se redujo en {abs(cambio_procesamiento_120):.2f}%.\n"
    else:
        informe += f"- **Aumento en el procesamiento**: La cantidad de ropa en mal estado procesada aumentó en {cambio_procesamiento_120:.2f}%.\n"
    
    if cambio_trabajadores_120 < 0:
        informe += f"- **Disminución en la contratación de personal**: A pesar del aumento en la demanda, el número total de trabajadores por boleta disminuyó en {abs(cambio_trabajadores_120):.2f}%.\n"
    else:
        informe += f"- **Aumento en la contratación de personal**: El número total de trabajadores por boleta aumentó en {cambio_trabajadores_120:.2f}%.\n"
        
    if cambio_demanda_insatisfecha_120 < 0:
        informe += f"- **Mejora en la satisfacción de demanda**: A pesar del aumento en la demanda, la demanda insatisfecha se redujo en {abs(cambio_demanda_insatisfecha_120):.2f}%, con un porcentaje de satisfacción global del {porcentaje_satisfecho_120:.2f}% (un cambio de {porcentaje_satisfecho_120 - porcentaje_satisfecho_base:.2f} puntos porcentuales respecto al escenario base).\n"
    else:
        informe += f"- **Deterioro en la satisfacción de demanda**: La demanda insatisfecha aumentó en {cambio_demanda_insatisfecha_120:.2f}%, con un porcentaje de satisfacción global del {porcentaje_satisfecho_120:.2f}% (un cambio de {porcentaje_satisfecho_120 - porcentaje_satisfecho_base:.2f} puntos porcentuales respecto al escenario base).\n"

    if cambio_costo_prenda_120 < 0:
        informe += f"- **Reducción en el costo por prenda**: El costo por prenda satisfecha se redujo en {abs(cambio_costo_prenda_120):.2f}%, lo que indica la presencia de economías de escala.\n\n"
    else:
        informe += f"- **Aumento en el costo por prenda**: El costo por prenda satisfecha aumentó en {cambio_costo_prenda_120:.2f}%, lo que indica la ausencia de economías de escala en el sistema.\n\n"
    
    # Elasticidad de costos
    elasticidad_80 = cambio_valor_objetivo_80 / -20  # -20% es el cambio en la demanda
    elasticidad_120 = cambio_valor_objetivo_120 / 20  # +20% es el cambio en la demanda
    
    informe += """### 3. Elasticidad de costos respecto a la demanda

"""
    
    informe += f"- **Elasticidad para demanda reducida**: {elasticidad_80:.4f} (un cambio del -20% en la demanda produce un cambio del {cambio_valor_objetivo_80:.2f}% en los costos)\n"
    informe += f"- **Elasticidad para demanda incrementada**: {elasticidad_120:.4f} (un cambio del +20% en la demanda produce un cambio del {cambio_valor_objetivo_120:.2f}% en los costos)\n\n"
    
    if elasticidad_80 < 1 and elasticidad_120 < 1:
        informe += "El sistema muestra **economías de escala** en ambos escenarios, ya que las elasticidades son menores a 1. Esto indica que los costos varían en menor proporción que la demanda.\n\n"
    elif elasticidad_80 > 1 and elasticidad_120 > 1:
        informe += "El sistema muestra **deseconomías de escala** en ambos escenarios, ya que las elasticidades son mayores a 1. Esto indica que los costos varían en mayor proporción que la demanda.\n\n"
    else:
        informe += "El sistema muestra un comportamiento mixto en cuanto a economías de escala, dependiendo de la dirección del cambio en la demanda.\n\n"
    
    # Robustez del modelo
    informe += """### 4. Robustez del modelo

"""
    
    # Evaluar la capacidad de absorción
    if cambio_valor_objetivo_120 < 40 and porcentaje_satisfecho_120 > 80:
        informe += "- **Alta capacidad de absorción**: El sistema muestra una buena capacidad para absorber incrementos en la demanda sin grandes aumentos en costos o reducciones significativas en la satisfacción.\n"
    elif cambio_valor_objetivo_120 > 40 or porcentaje_satisfecho_120 < 70:
        informe += "- **Baja capacidad de absorción**: El sistema muestra dificultades para absorber incrementos en la demanda, resultando en aumentos significativos de costos o reducciones importantes en la satisfacción.\n"
    else:
        informe += "- **Capacidad de absorción moderada**: El sistema puede absorber incrementos moderados en la demanda, aunque con ciertos aumentos en costos o reducciones en la satisfacción.\n"
    
    # Evaluar la flexibilidad operativa
    if abs(cambio_trabajadores_80) > 15 or abs(cambio_trabajadores_120) > 15:
        informe += "- **Alta flexibilidad operativa**: El sistema muestra una buena capacidad para ajustar sus recursos humanos ante variaciones en la demanda.\n"
    else:
        informe += "- **Baja flexibilidad operativa**: El sistema muestra limitaciones para ajustar sus recursos humanos ante variaciones en la demanda.\n"

    # Conclusiones
    informe += """
## Conclusiones

"""
    
    # Conclusión sobre escenario de demanda reducida
    informe += """1. **Escenario de demanda reducida (80%)**:
"""
    
    if cambio_valor_objetivo_80 < -15:
        informe += f"   - Resulta en una reducción significativa de costos ({abs(cambio_valor_objetivo_80):.2f}%), "
    elif cambio_valor_objetivo_80 < 0:
        informe += f"   - Resulta en una leve reducción de costos ({abs(cambio_valor_objetivo_80):.2f}%), "
    else:
        informe += f"   - No logra reducir los costos y de hecho los aumenta en un {cambio_valor_objetivo_80:.2f}%, "
        
    if porcentaje_satisfecho_80 > porcentaje_satisfecho_base:
        informe += f"mejor satisfacción de la demanda ({porcentaje_satisfecho_80:.2f}% vs {porcentaje_satisfecho_base:.2f}% en el escenario base) "
    else:
        informe += f"peor satisfacción de la demanda ({porcentaje_satisfecho_80:.2f}% vs {porcentaje_satisfecho_base:.2f}% en el escenario base) "
        
    if util_rh_prom_80 < util_rh_prom_base:
        informe += f"y menor utilización de recursos humanos ({util_rh_prom_80:.2f}% vs {util_rh_prom_base:.2f}%).\n"
    else:
        informe += f"y mayor utilización de recursos humanos ({util_rh_prom_80:.2f}% vs {util_rh_prom_base:.2f}%).\n"
    
    if cambio_costo_prenda_80 > 0:
        informe += f"   - El costo por prenda satisfecha es mayor (${costo_prenda_80:.2f} vs ${costo_prenda_base:.2f}), lo que indica la presencia de costos fijos significativos en el sistema.\n\n"
    else:
        informe += f"   - El costo por prenda satisfecha es menor (${costo_prenda_80:.2f} vs ${costo_prenda_base:.2f}), lo que sugiere ausencia de economías de escala en este rango de operación.\n\n"
    
    # Conclusión sobre escenario de demanda incrementada
    informe += """2. **Escenario de demanda incrementada (120%)**:
"""
    
    if cambio_valor_objetivo_120 > 15:
        informe += f"   - Resulta en un aumento significativo de costos ({cambio_valor_objetivo_120:.2f}%), "
    elif cambio_valor_objetivo_120 > 0:
        informe += f"   - Resulta en un leve aumento de costos ({cambio_valor_objetivo_120:.2f}%), "
    else:
        informe += f"   - Sorprendentemente reduce los costos en un {abs(cambio_valor_objetivo_120):.2f}%, "
        
    if porcentaje_satisfecho_120 < porcentaje_satisfecho_base:
        informe += f"menor satisfacción de la demanda ({porcentaje_satisfecho_120:.2f}% vs {porcentaje_satisfecho_base:.2f}% en el escenario base) "
    else:
        informe += f"mejor satisfacción de la demanda ({porcentaje_satisfecho_120:.2f}% vs {porcentaje_satisfecho_base:.2f}% en el escenario base) "
        
    if util_rh_prom_120 > util_rh_prom_base:
        informe += f"y mayor utilización de recursos humanos ({util_rh_prom_120:.2f}% vs {util_rh_prom_base:.2f}%).\n"
    else:
        informe += f"y menor utilización de recursos humanos ({util_rh_prom_120:.2f}% vs {util_rh_prom_base:.2f}%).\n"
    
    if cambio_costo_prenda_120 < 0:
        informe += f"   - El costo por prenda satisfecha es menor (${costo_prenda_120:.2f} vs ${costo_prenda_base:.2f}), lo que indica la presencia de economías de escala en el sistema.\n\n"
    else:
        informe += f"   - El costo por prenda satisfecha es mayor (${costo_prenda_120:.2f} vs ${costo_prenda_base:.2f}), lo que sugiere deseconomías de escala en este rango de operación.\n\n"
    
    # Implicaciones para la planificación
    informe += """3. **Implicaciones para la planificación**:

   - **Flexibilidad operativa**: La fundación debe desarrollar mecanismos de flexibilidad para ajustar sus operaciones ante variaciones en la demanda, especialmente en lo referente a la contratación de personal por boleta.
   
   - **Capacidad de reserva**: Es recomendable mantener cierta capacidad de reserva para absorber potenciales incrementos en la demanda, particularmente si estos pueden generar altos costos por demanda insatisfecha.
   
   - **Gestión de costos fijos**: Evaluar estrategias para reducir los costos fijos y mejorar la eficiencia operativa ante posibles reducciones en la demanda, lo que permitiría mantener un costo por prenda más competitivo incluso con volúmenes menores.
   
   - **Optimización de inventarios**: """
   
    if (util_cap_prom_120 > util_cap_prom_base) and (util_cap_prom_80 > util_cap_prom_base):
        informe += "Los diferentes escenarios muestran un aumento en la utilización del almacenamiento, lo que sugiere la necesidad de revisar las políticas de inventario para asegurar su eficiencia con diferentes niveles de demanda.\n"
    else:
        informe += "Las políticas de gestión de inventarios parecen responder adecuadamente a las variaciones de demanda, aunque podría ser útil revisar la distribución por tipo de inventario para optimizar aún más su uso.\n"
    
    # Guardar el informe en un archivo Markdown
    os.makedirs(RESPUESTAS_DIR, exist_ok=True)
    with open(RESPUESTAS_DIR / f'pregunta6_sensibilidad_consolidado.md', 'w') as f:
        f.write(informe)
    
    return informe

if __name__ == "__main__":
    print("Resolviendo el modelo de optimización para la pregunta 6...")
    
    # Asegurarse que exista la carpeta para los resultados
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Resolver los tres modelos
    print("Resolviendo caso base...")
    resultados_base = resolver_modelo_base()
    
    print("Resolviendo caso con demanda al 80%...")
    resultados_demanda_80 = resolver_modelo_demanda_variable(0.8)
    
    print("Resolviendo caso con demanda al 120%...")
    resultados_demanda_120 = resolver_modelo_demanda_variable(1.2)
    
    if resultados_base is not None and resultados_demanda_80 is not None and resultados_demanda_120 is not None:
        print(f"\nModelos resueltos correctamente.")
        print(f"Valor óptimo caso base: ${resultados_base[0]['valor_objetivo']:,.2f}")
        print(f"Valor óptimo con demanda al 80%: ${resultados_demanda_80[0]['valor_objetivo']:,.2f}")
        print(f"Valor óptimo con demanda al 120%: ${resultados_demanda_120[0]['valor_objetivo']:,.2f}")
        
        # Generar gráficos comparativos
        print("\nGenerando gráficos comparativos...")
        generar_graficos_comparativos(resultados_base, resultados_demanda_80, resultados_demanda_120)
        
        # Generar informe consolidado
        print("\nGenerando informe consolidado...")
        informe_consolidado = generar_informe_consolidado(resultados_base, resultados_demanda_80, resultados_demanda_120)
        
        print(f"\nSe han generado los gráficos comparativos en: {OUTPUT_DIR}")
        print(f"Se ha generado el informe consolidado en: {RESPUESTAS_DIR}/pregunta6_sensibilidad_consolidado.md")
    else:
        print("No se pudieron resolver los modelos. Verifique los datos y restricciones.")

