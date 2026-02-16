import re
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import urllib.request
import re

def screen_resolution_split(df):

    # Posibles valores después de examinar el dataset entero laptops.csv
    features_resolution = ["Full HD", "IPS Panel", "Touchscreen", "4K Ultra HD", "Retina Display", "Quad HD+"]

    # Crear una columna binaria para cada tipo de resolución
    for feature in features_resolution:
        col_name = feature.replace(" ", "_").replace("+", "plus").replace("-", "_").replace(",", "").replace("/", "_")
        df[col_name] = df["ScreenResolution"].apply(lambda x: int(feature in str(x)))

    # Limpiar la columna para que solo quede numxnum (si existe)
    df["ScreenResolution"] = df["ScreenResolution"].str.extract(r"(\d{3,5}x\d{3,5})")

    # Extraer ancho y alto de la resolución
    df['Resolution_width'] = df['ScreenResolution'].str.extract(r'(\d+)x')[0].astype(int)
    df['Resolution_height'] = df['ScreenResolution'].str.extract(r'x(\d+)')[0].astype(int)
    df['PPI'] = np.sqrt(df['Resolution_width']**2 + df['Resolution_height']**2) / df['Inches']

    df.drop(["ScreenResolution", 'Resolution_width', 'Resolution_height', 'Inches'], axis=1, inplace=True)
    
def replace_value(df, col, old, new, new_type=int):

    # Nueva columna con el mismo nombre pero añadiendo el valor que se ha reemplazado
    new_col = col + "_" + old

    # Hacemos el reemplazo y convertimos a un nuevo tipo de dato
    df[new_col] = df[col].str.replace(old, new).astype(new_type)

    # Eliminamos la columna original
    df.drop(col, axis=1, inplace=True)

def memory_split(df):
    
    features_memory = ["SSD", "HDD", "Flash Storage", "Hybrid"]

    # Crear una columna binaria para cada tipo de memoria
    for feature in features_memory:
        col_name = feature.replace(" ", "_").replace("+", "plus").replace("-", "_").replace(",", "").replace("/", "_")
        df[col_name] = df["Memory"].apply(lambda x: int(feature in str(x)))

    # Separar la memoria en dos columnas si hay un +
    df[["Memory1", "Memory2"]] = df["Memory"].str.split("+", expand=True).astype(str)

    df["Memory_GB"] = 0

    for i in ["1", "2"]:
        col = "Memory" + i
        def extract_gb(x):
            if pd.isnull(x):
                return 0
            s = str(x)
            # Elimina el tipo de memoria si está presente
            for f in features_memory:
                s = s.replace(f, "")
            s = s.strip()
            if "GB" in s:
                return int(s.replace("GB", "").strip())
            elif "TB" in s:
                return int(float(s.replace("TB", "").strip()) * 1024)
            elif s.isdigit():
                return int(s)
            else:
                return 0
        df["Memory_GB"] = df["Memory_GB"] + df[col].apply(extract_gb)

    df.drop(["Memory", "Memory1", "Memory2"], axis=1, inplace=True)

    # Llenar las nuevas columnas con el tipo de memoria correspondiente, 
    # eliminando el nombre del tipo de memoria y dejando solo la capacidad
    # Eliminamos las unidades de GB y TB, y convertios a un tipo numérico, asegurándonos de convertir TB a GB (1 TB = 1024 GB)
    # for feature in features_memory:
    #     for i in ["1", "2"]:
    #         col = feature + i
    #         df[col] = df["Memory" + i].apply(lambda x: x.replace(feature, "").strip() if pd.notnull(x) and feature in x else np.nan)
    #         df[col] = df[col].apply(lambda x: int(x.replace("GB", "").strip()) if pd.notnull(x) and "GB" in str(x) \
    #                 else (int(float(x.replace("TB", "").strip()) * 1024) if pd.notnull(x) and "TB" in str(x) else 0))
    #     feature_gb = feature + "_GB"
    #     df[feature_gb] = df[feature + "1"] + df[feature + "2"]

    # # Crear columna para marcar filas donde no hay ni SSD ni HDD usando la columna original 'Memory'
    # mask_no_ssd_hdd = ~df["Memory"].str.contains("SSD") & ~df["Memory"].str.contains("HDD")

    # # Extraer valores numGB o numTB de la columna original 'Memory'
    # def extract_gb_tb(val):
    #     import re
    #     if not isinstance(val, str):
    #         return 0
    #     match_gb = re.search(r"(\d+(?:\.\d+)?)GB", val)
    #     match_tb = re.search(r"(\d+(?:\.\d+)?)TB", val)
    #     gb = float(match_gb.group(1)) if match_gb else 0
    #     tb = float(match_tb.group(1)) * 1024 if match_tb else 0
    #     return int(gb + tb)

    # df["Memory_no_SSD_HDD"] = 0
    # df.loc[mask_no_ssd_hdd, "Memory_no_SSD_HDD"] = df.loc[mask_no_ssd_hdd, "Memory"].apply(extract_gb_tb)

    #df["Memory_GB_TB"] = df["Memory"].apply(extract_gb_tb)

    # Eliminar las columnas originales de memoria
    #df.drop(["Memory", "Memory1", "Memory2"] + [f"{feature}{i}" for feature in features_memory for i in ["1", "2"]], axis=1, inplace=True)

def memory_split2(df):
    
    df["Memory_GB"] = 0
    #df["SSD"] = df["HDD"] = df["Flash Storage"] = df["Hybrid"] = 0

    features_memory = ["HDD", "Hybrid", "SSD", "Flash Storage"]
    
    # Procesa cada fila
    for idx, value in df["Memory"].fillna("").items():

        parts = [p.strip() for p in value.split("+")]

        for part in parts:
               
            for feature in features_memory:        
            
                if feature in part:
                    if "TB" in part:
                        qty = float(part.replace(feature, "").replace("TB", "").replace("GB", "").strip()) * 1024
                    else:
                        qty = float(part.replace(feature, "").replace("GB", "").replace("TB", "").strip())

                    df.at[idx, "Memory_GB"] += int(qty)
                    df.at[idx, "Memory_Type"] = feature #1
        
        
    # Eliminar las columnas originales de memoria
    df.drop("Memory", axis=1, inplace=True)

def cpu_split(df):

    # Separamos la CPU en 2 columnas: una con la marca del procesador y otra con la velocidad en GHz
    df["Cpu_Brand"] = df["Cpu"].str.extract(r'([A-Za-z]+)').iloc[:, 0]
    df["Speed_GHz"] = df["Cpu"].str.extract(r'(\d+\.?\d*)GHz').astype(float)

    # Eliminamos la columna original de CPU
    df.drop("Cpu", axis=1, inplace=True)

def cpu_split2(df):

    # Separamos la CPU en 2 columnas: una con la marca del procesador y otra con la velocidad en GHz
    df["Speed_GHz"] = df["Cpu"].str.extract(r'(\d+\.?\d*)GHz').astype(float)    
    df["Cpu_Brand"] = df["Cpu"].apply(extract_cpu_brand).astype(str) 
    #df["Cpu_Family"] = df["Cpu"].apply(extract_cpu_family).astype(str) 

    # Eliminamos la columna original de CPU
    df.drop("Cpu", axis=1, inplace=True)

def procesar_cpu_caotico(df, columna_texto):

    import re

    # 1. Diccionario de búsqueda
    dict_cpu = {
        'Intel': ['i3', 'i5', 'i7', 'i9', 'core ultra', 'xeon', 'pentium', 'celeron', 'atom'],
        'AMD': ['ryzen 3', 'ryzen 5', 'ryzen 7', 'ryzen 9', 'threadripper', 'epyc', 'athlon', 'sempron'],
        'Apple': ['m1', 'm2', 'm3', 'm4'],
        'Qualcomm': ['snapdragon', 'elite x']
    }

    def extraer_info(texto):
        if not isinstance(texto, str):
            return "Other", "Other", None
        
        texto_clean = texto.lower()
        marca_encontrada = "Other"
        familia_encontrada = "Other"
        
        # Extraer Marca y Familia
        for marca, familias in dict_cpu.items():
            if marca.lower() in texto_clean:
                marca_encontrada = marca
            for familia in familias:
                if familia in texto_clean:
                    familia_encontrada = familia.upper()
                    # Si encontramos la familia, a veces podemos deducir la marca (ej: Ryzen -> AMD)
                    if marca_encontrada == "Other":
                        marca_encontrada = marca
                    break
        
        # Extraer Velocidad (GHz) usando Regex
        # Busca un número (con punto o coma) seguido de ghz
        match_ghz = re.search(r'(\d+[.,]\d+|\d+)\s*(?:ghz)', texto_clean)
        velocidad = float(match_ghz.group(1).replace(',', '.')) if match_ghz else None
        
        return marca_encontrada, familia_encontrada, velocidad
    # Aplicar la función y expandir a nuevas columnas
    df[['cpu_marca', 'cpu_familia', 'cpu_ghz']] = df[columna_texto].apply(
        lambda x: pd.Series(extraer_info(x))
    )

    # Eliminamos la columna original de CPU
    df.drop("Cpu", axis=1, inplace=True)    

    return df

def extract_cpu_brand(cpu_str):

    cpu_str = str(cpu_str)
   
    return " ".join(cpu_str.split()[:1])
    
def extract_cpu_family(cpu_str):

    cpu_str = str(cpu_str)
    cpu_upper = cpu_str.upper()
    
    if cpu_upper.startswith("AMD"):
        return " ".join(cpu_str.split()[:2])
    elif any(x in cpu_upper for x in ["XEON", "ATOM"]) and "INTEL" in cpu_upper:
        return " ".join(cpu_str.split()[:2])
    elif cpu_upper.startswith("INTEL"):
        return " ".join(cpu_str.split()[:3])
    else:
        return "Other"
    
def gpu_split(df):

    # Separamos la GPU en una columna con la marca del fabricante
    df["Gpu_Brand"] = df["Gpu"].str.extract(r'([A-Za-z]+)').iloc[:, 0]

    # Eliminamos la columna original de GPU
    df.drop("Gpu", axis=1, inplace=True)
    

def gpu_split2(df):    

    df['Gpu_Brand'] = df['Gpu'].apply(extract_gpu_brand)
    #df['Gpu_Family'] = df['Gpu'].apply(extract_gpu_family)

    # Eliminamos la columna original de GPU
    df.drop("Gpu", axis=1, inplace=True)

def procesar_gpu_caotico(df, columna_texto):
    # 1. Diccionario robusto de GPUs
    dict_gpu = {
        'Nvidia': ['rtx', 'gtx', 'quadro', 'geforce', 'turing', 'pascal', 'mx'],
        'AMD': ['radeon', 'rx', 'vega', 'rdna', 'firepro'],
        'Intel': ['iris', 'arc', 'uhd', 'graphics', 'hd graphics'],
        'Apple': ['m1', 'm2', 'm3', 'm4'] # En Apple, CPU y GPU suelen ir juntas
        }

    def extraer_info_gpu(texto):
        if not isinstance(texto, str): return "Other", "Other", None

        texto_clean = texto.lower()
        marca_encontrada = "Other"
        familia_encontrada = "Other"
 
        # Extraer Marca y Familia
        for marca, palabras_clave in dict_gpu.items():
            for palabra in palabras_clave:
                if palabra in texto_clean:
                    marca_encontrada = marca
                    # Intentamos capturar el modelo específico (ej: rtx 3060)
                    match_modelo = re.search(fr'({palabra}\s*\d+)', texto_clean)
                    familia_encontrada = match_modelo.group(1).upper() if match_modelo else palabra.upper()
                    break
            if marca_encontrada != "Other": 
                break
                   
        return marca_encontrada, familia_encontrada

    # Aplicar y expandir
    df[['gpu_marca', 'gpu_modelo']] = df[columna_texto].apply(
    lambda x: pd.Series(extraer_info_gpu(x))
    )

    # Eliminamos la columna original de GPU
    df.drop("Gpu", axis=1, inplace=True)

    return df

def extract_gpu_brand(gpu_name):

    gpu_str = str(gpu_name)

    return " ".join(gpu_str.split()[:1])

def extract_gpu_family(gpu_name):
    
    gpu_str = str(gpu_name)
    
    return " ".join(gpu_str.split()[:2])

# Función para separar sistema operativo y versión
def ordenar_cpu_familia(fam_string):
    up_str = str(fam_string).upper()
    # Más baratos
    if "ATOM" in up_str:
        fam_sort = 1
    elif "CELERON" in up_str:
        fam_sort = 2
    elif "PENTIUM" in up_str:
        fam_sort = 3
    elif "I3" in up_str:
        fam_sort = 4
    elif "I5" in up_str:
        fam_sort = 5
    elif "I7" in up_str:
        fam_sort = 6
    elif "I9" in up_str:
        fam_sort = 7
    elif "XEON" in up_str:
        fam_sort = 8
    else:
        fam_sort = 1
    return int(fam_sort)

def split_cpu_familia(df):

    df["cpu_familia"] = df["cpu_familia"].apply(lambda x: pd.Series(ordenar_cpu_familia(x))).astype(int)

# Función para separar sistema operativo y versión
def ordenar_OpSys(os_string):

    up_str = str(os_string).upper()
    os_vers = 0
    parts = up_str.split(" ", 1)
    if len(parts) > 1:
        try:
            os_vers = float(parts[1]) if parts[1].replace('.', '', 1).isdigit() else 0
        except (ValueError, TypeError):
            os_vers = 0

    # Más baratos
    if "NO OS" in up_str:
        os_sort = 1
    # Linux 
    elif "LINUX" in up_str:
        os_sort = 2
    # Windows
    elif "WINDOWS" in up_str:
        os_sort = 3
    # Mac OS X y macOS
    elif "MAC" in up_str:
        os_sort = 100
    else:
        os_sort = 1

    return int(os_sort), int(os_vers)

def split_OpSys(df):

    df[["Sistema", "Version"]] = df["OpSys"].apply(lambda x: pd.Series(ordenar_OpSys(x)))

    # Eliminamos la columna original
    df.drop("OpSys", axis=1, inplace=True)

def other_grouping(df, col, threshold=10):

    # Agrupar las categorías con menos de un umbral de muestras en "Other"
    counts = df[col].value_counts()
    to_replace = counts[counts < threshold].index
    df[col] = df[col].replace(to_replace, "Other")

def other_grouping2(df, cols, new_col, threshold=10):
    
    df[new_col] = 0
    
    for col in cols:
        counts = df[col].sum()
        if counts < threshold:
            df[new_col] = df[new_col] + df[col]
            df.drop(col, axis=1, inplace=True)
        

def target_encoding(df, df_mean, cols, target_col):

    # Para cada columna en cols, calculamos el valor medio de target_col para cada categoría 
    # y lo asignamos a una nueva columna con el nombre original más "_Encoded"
    for col in cols:
        target_mean = df_mean.groupby(col)[target_col].median()
        df[col+"_Encoded"] = df[col].map(target_mean)
        #df[col+"_Encoded"].fillna(df_mean[target_col].mean(), inplace=True)
        df[col+"_Encoded"] = df[col+"_Encoded"].fillna(df_mean[target_col].median())

    # Eliminamos las columnas originales
    df.drop(cols, axis=1, inplace=True)

def frequency_encoding(df, df_train, cols):

    # Para cada columna en cols, calculamos la frecuencia de cada categoría 
    # y lo asignamos a una nueva columna con el nombre original más "_Freq"
    for col in cols:
        freq = df_train[col].value_counts(normalize=True)
        df[col+"_Freq"] = df[col].map(freq)

    # Eliminamos las columnas originales
    df.drop(cols, axis=1, inplace=True)

def one_hot_encoding(df, cols):

    # Hacemos un one-hot encoding para las columnas especificadas en cols, 
    # eliminando la primera categoría para evitar la multicolinealidad
    df = pd.get_dummies(df, columns=cols, drop_first=True, dtype=int)

    return df

def outlier_iqr(df, col):

    # Calcular el rango intercuartílico (IQR) para la columna especificada
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    # Definir los límites inferior y superior para detectar outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df[col] > upper_bound) | (df[col] < lower_bound)]
    print(f"\nOutliers in {col}: \n{outliers[[col]].reset_index()}")
    
def gpu_split3(df):

    # Aplicar funciones
    df['brand_gpu'] = df['Gpu'].apply(get_brand)
    df['family_gpu'] = df['Gpu'].apply(get_family)
    #df['model_number_gpu'] = df['Gpu'].apply(get_model_number)

    # Eliminamos la columna original de GPU
    df.drop("Gpu", axis=1, inplace=True)


# Función para extraer marca
def get_brand(name):
    if 'intel' in name.lower():
        return 'Intel'
    elif 'amd' in name.lower():
        return 'AMD'
    elif 'nvidia' in name.lower():
        return 'Nvidia'
    else:
        return 'Other'

# Función para extraer familia
familias = ['Iris', 'HD Graphics', 'UHD Graphics', 'Radeon', 'GeForce', 'FirePro', 'Quadro']
def get_family(name):
    for fam in familias:
        if fam.lower() in name.lower():
            return fam
    return 'Other'

# Función para extraer número de modelo (primer número de 3-4 dígitos)
def get_model_number(name):
    match = re.search(r'(\d{3,4})', name)
    if match:
        return match.group(1)
    return None


def categorizar_procesador(cpu):
    cpu = cpu.lower()
    if 'i9' in cpu or 'ryzen 9' in cpu or 'threadripper' in cpu:
        return 4  # Entusiasta
    elif 'i7' in cpu or 'ryzen 7' in cpu:
        return 3  # Alta
    elif 'i5' in cpu or 'ryzen 5' in cpu:
        return 2  # Media
    elif 'i3' in cpu or 'ryzen 3' in cpu or 'celeron' in cpu or 'pentium' in cpu:
        return 1  # Entrada
    else:
        return 0  # Otros / Antiguos