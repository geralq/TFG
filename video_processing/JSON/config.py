import os
from absl import logging

# Configuraciones globales
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.set_verbosity(logging.ERROR)

# Directorios
OUTPUT_PATH = '/home/gerardo/LSE_DATABASE/LSE_HEALTH'
VIDEO_BASE_PATH = '/home/gerardo/LSE_HEALTH/Videos-LSE-Health-UVigo/Videos-LSE-Health-UVigo'
EXCEL_PATH = '/home/gerardo/LSE_HEALTH/LSE-Health-UVigo.xlsx'

# Crear directorio si no existe
os.makedirs(OUTPUT_PATH, exist_ok=True)
