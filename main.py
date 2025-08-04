# Entry point for training/testing pipeline

from preprocessing.preprocessing_pipeline import run_preprocessing_pipeline

if __name__ == "__main__":
    import pandas as pd
    data = pd.read_csv('messy_bank_transactions.csv')
    df = pd.DataFrame(data)
    from models.anomaly_detector import detect_anomalies_with_isolation_forest
    
    data, reason = run_preprocessing_pipeline(df)
    if reason is None:
        y, z = detect_anomalies_with_isolation_forest(data)
        print(y[y['anomaly'] == -1].sample(5))

    
