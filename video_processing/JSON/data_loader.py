import pandas as pd
import ast
from JSON.config import EXCEL_PATH

def load_signer_video(path="signer_video.xlsx"):
    signer_video = pd.read_excel(path)
    signer_video['ELAN file'] = signer_video['ELAN file'].apply(ast.literal_eval)
    return signer_video

def load_signer_bow(path="signer_bow.xlsx"):
    return pd.read_excel(path)

def load_glosses_content():
    dfs = pd.read_excel(EXCEL_PATH, sheet_name=None)
    return dfs['GlossesContent'], dfs
