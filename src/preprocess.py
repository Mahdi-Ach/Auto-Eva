import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

def load_and_preprocess(path, test_size=0.2, random_state=42):
    df = pd.read_csv(path)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna(subset=['TotalCharges'])
    df = df.drop('customerID', axis=1)
    
    cat_cols = df.select_dtypes(include=['object', 'string']).columns.drop('Churn')
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    
    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), cat_cols),
        ('num', 'passthrough', num_cols)
    ])
    
    X_raw = df.drop('Churn', axis=1)
    y = (df['Churn'] == 'Yes').astype(int)
    
    X = preprocessor.fit_transform(X_raw)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    
    return X_train, X_val, y_train, y_val, preprocessor