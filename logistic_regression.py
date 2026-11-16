import os
import time
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix

from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# EL CODIGO QUE ESTE ESCRITO

# -----------------
# 1. CONFIGURACION  
# -----------------
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

RUTA_DATASET = "datos/train_clean/multilingual_en.parquet" 
RUTA_TWEETFEELS = "datos/tweetfeels.parquet"
RUTA_TOXITEXT = "datos/toxi_text.parquet"
DIR_OUT = "resultados"
os.makedirs(DIR_OUT, exist_ok=True)

COL_TEXTO = "text"     
COL_TARGET = "label"   # Variable objetivo principal: sentimiento
COLS_META = ["domain", "source"]


# ------------------------------------------------------------------------
# 2. CARGA Y PARTICION UNICA (LOS INDICES SE COMPARTEN CON LA GRU Y CON JON)
# -----------------------------------------------------------------------
