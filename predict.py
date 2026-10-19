import joblib
import numpy as np
import pandas as pd
import re

def engineer_features(data):
    df_feat = data.copy()
    
    # 1. Type mapping
    type_map = {'L': 0, 'M': 1, 'H': 2}
    if 'Type' in df_feat.columns:
        df_feat['Type_num'] = df_feat['Type'].map(type_map).fillna(0)
    
    # 2. Temperature difference
    if 'Process temperature [K]' in df_feat.columns and 'Air temperature [K]' in df_feat.columns:
        df_feat['temp_diff'] = df_feat['Process temperature [K]'] - df_feat['Air temperature [K]']
    
    # 3. Mechanical Power in Watts: Torque * Angular Velocity
    if 'Rotational speed [rpm]' in df_feat.columns and 'Torque [Nm]' in df_feat.columns:
        df_feat['rotational_rad_s'] = df_feat['Rotational speed [rpm]'] * (2 * np.pi / 60)
        df_feat['power_W'] = df_feat['Torque [Nm]'] * df_feat['rotational_rad_s']
        # 4. Overstrain
        df_feat['overstrain'] = df_feat['Torque [Nm]'] * df_feat['Tool wear [min]']
    
    # Drop non-feature and target columns if present
    drop_cols = ['UDI', 'Product ID', 'Type', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF', 'Machine failure']
    for col in drop_cols:
        if col in df_feat.columns:
            df_feat = df_feat.drop(columns=[col])
            
    # Clean brackets [ ] from column names to match model training
    df_feat.columns = [re.sub(r'[\[\]<>,:]', '_', col) for col in df_feat.columns]
    
    return df_feat

def predict(input_df: pd.DataFrame) -> np.ndarray:
    """
    Accepts an input DataFrame (matching train.csv or test.csv)
    and returns an array of binary failure predictions (0 or 1).
    """
    # Load the saved model artifact
    artifact = joblib.load('models/trained_pipeline.joblib')
    model = artifact['model']
    threshold = artifact['optimal_threshold']
    feature_names = artifact['feature_names']
    
    # Apply feature transformations
    X_processed = engineer_features(input_df)
    
    # Align exact column order
    X_aligned = X_processed[feature_names]
    
    # Predict probabilities and apply optimal threshold
    probabilities = model.predict_proba(X_aligned)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    
    return predictions

if __name__ == '__main__':
    # Self-test: run on a small sample of the data to verify
    sample_df = pd.read_csv('data/train.csv').head(10)
    test_preds = predict(sample_df)
    print("Inference Test Successful!")
    print(f"Sample Predictions (0=Healthy, 1=Fail): {test_preds}")