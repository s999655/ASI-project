
import pandas as pd
from typing import Tuple


def preprocess_data(df: pd.DataFrame, has_target: bool = True) -> Tuple[pd.DataFrame, pd.Series | None]:
    
    df = df.copy()
    
    # Extract target if present
    target = None
    if has_target:
        if 'Attrition' not in df.columns:
            raise ValueError("Target column 'Attrition' not found in data")
        target_map = {'Yes': 1, 'No': 0}
        target = df["Attrition"].apply(lambda x: target_map[x])
        df = df.drop(['Attrition'], axis=1)
    
    # Drop Attrition_numerical if it exists
    if 'Attrition_numerical' in df.columns:
        df = df.drop(['Attrition_numerical'], axis=1)
    
    # Separate categorical and numerical columns
    categorical = []
    for col, value in df.items():
        if value.dtype == 'object':
            categorical.append(col)
    
    numerical = df.columns.difference(categorical)
    
    # One-hot encode categorical variables
    df_cat = df[categorical]
    df_cat = pd.get_dummies(df_cat)
    
    # Keep numerical variables
    df_num = df[numerical]
    
    # Concatenate numerical and encoded categorical
    df_final = pd.concat([df_num, df_cat], axis=1)
    
    return df_final, target


def align_features(df: pd.DataFrame, reference_columns: list) -> pd.DataFrame:
    
    # Add missing columns with zeros
    for col in reference_columns:
        if col not in df.columns:
            df[col] = 0
    
    # Keep only columns that exist in reference
    df = df[reference_columns]
    
    return df
