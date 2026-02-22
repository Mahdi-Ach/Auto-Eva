import xgboost as xgb
from preprocess import load_and_preprocess  
import joblib

if __name__ == "__main__":
    DATA_PATH = "./dataset/Telco-Customer-Churn.csv" 
    
    print("Loading & preprocessing data...")
    X_train, X_val, y_train, y_val, preprocessor = load_and_preprocess(DATA_PATH)
    
    print(f"Train shape: {X_train.shape} | Val shape: {X_val.shape}")
    
    scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval   = xgb.DMatrix(X_val, label=y_val)
    
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'aucpr', 
        'eta': 0.05,            
        'max_depth': 5,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'scale_pos_weight': scale_pos_weight,
        'lambda': 1.0,    
        'seed': 42,
        'tree_method': 'hist',  
        'device': 'cpu'               
    }
    
    print("Training model...")
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        evals=[(dtrain, 'train'), (dval, 'val')],
        early_stopping_rounds=15,
        verbose_eval=50
    )
    
    model.save_model('../models/model_vfinal.json')
    print(f"Best iteration: {model.best_iteration}")
    joblib.dump(preprocessor, '../models/preprocessor.joblib')
    print("Preprocessor saved to models/preprocessor.joblib")