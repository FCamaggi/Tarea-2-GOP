#!/usr/bin/env python3

import os
import sys
import shutil
import argparse
import pandas as pd
import glob
from pathlib import Path

def create_directory_structure():
    """Crea la estructura de directorios para las partes y preguntas"""
    
    # Partes principales
    partes = ['ParteA', 'ParteB', 'ParteC']
    
    # Número de preguntas por parte (ajusta según sea necesario)
    num_preguntas = {
        'ParteA': 6,
        'ParteB': 3,
        'ParteC': 3
    }
    
    latex_dir = Path('latex')
    
    # Crear directorios por parte
    for parte in partes:
        parte_dir = latex_dir / parte
        resources_dir = parte_dir / 'resources'
        
        # Crear directorio principal de parte si no existe
        os.makedirs(parte_dir, exist_ok=True)
        
        # Crear directorio resources si no existe
        os.makedirs(resources_dir, exist_ok=True)
        
        # Crear subdirectorios para cada pregunta
        for i in range(1, num_preguntas[parte] + 1):
            pregunta_dir = resources_dir / f'pregunta{i}'
            os.makedirs(pregunta_dir, exist_ok=True)
            print(f"Creado directorio: {pregunta_dir}")
    
    print("Estructura de directorios creada exitosamente.")

def copy_csv_to_structure(desarrollo_dir='desarrollo'):
    """Copia archivos CSV y gráficos desde desarrollo a la estructura LaTeX"""
    
    latex_dir = Path('latex')
    
    # Mapea las carpetas de desarrollo a las carpetas de LaTeX
    mappings = {
        'src/pregunta2': 'ParteA/resources/pregunta2',
        'src/pregunta3': 'ParteA/resources/pregunta3',
        'src/pregunta4': 'ParteA/resources/pregunta4',
        'src/pregunta5': 'ParteA/resources/pregunta5',
        'src/pregunta6': 'ParteA/resources/pregunta6',
    }
    
    desarrollo_path = Path(desarrollo_dir)
    
    for src_rel, dest_rel in mappings.items():
        src_path = desarrollo_path / src_rel
        dest_path = latex_dir / dest_rel
        
        # Asegúrate de que el directorio destino existe
        os.makedirs(dest_path, exist_ok=True)
        
        # Copia archivos CSV
        for csv_file in src_path.glob('*.csv'):
            dest_file = dest_path / csv_file.name
            shutil.copy2(csv_file, dest_file)
            print(f"Copiado: {csv_file} -> {dest_file}")
        
        # Copia archivos de gráficos
        for img_file in src_path.glob('*.png'):
            dest_file = dest_path / img_file.name
            shutil.copy2(img_file, dest_file)
            print(f"Copiado: {img_file} -> {dest_file}")
    
    print("Archivos CSV y gráficos copiados exitosamente.")

def main():
    parser = argparse.ArgumentParser(description='Organiza la estructura de archivos para el proyecto LaTeX')
    parser.add_argument('--create-dirs', action='store_true', help='Crear estructura de directorios')
    parser.add_argument('--copy-files', action='store_true', help='Copiar archivos desde desarrollo')
    
    args = parser.parse_args()
    
    if args.create_dirs:
        create_directory_structure()
    
    if args.copy_files:
        copy_csv_to_structure()
    
    if not (args.create_dirs or args.copy_files):
        parser.print_help()

if __name__ == "__main__":
    main()
