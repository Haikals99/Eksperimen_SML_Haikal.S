import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

def load_data(filepath):
    """Memuat dataset diabetes dari file CSV."""
    print("Loading dataset...")
    df = pd.read_csv(filepath)
    print(f"Dataset berhasil dimuat: {df.shape[0]} baris, {df.shape[1]} kolom")
    return df

def remove_duplicates(df):
    """Menghapus data duplikat."""
    before = df.shape[0]
    df = df.drop_duplicates()
    after = df.shape[0]
    print(f"Duplikat dihapus: {before - after} baris")
    return df

def handle_missing_values(df):
    """Menangani missing values."""
    missing = df.isnull().sum().sum()
    print(f"Missing values ditemukan: {missing}")
    df = df.dropna()
    print(f"Shape setelah handle missing values: {df.shape}")
    return df

def remove_outliers(df):
    """Menghapus outlier pada kolom BMI menggunakan metode IQR."""
    before = df.shape[0]
    Q1 = df['BMI'].quantile(0.25)
    Q3 = df['BMI'].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df = df[(df['BMI'] >= lower) & (df['BMI'] <= upper)]
    after = df.shape[0]
    print(f"Outlier BMI dihapus: {before - after} baris")
    print(f"Batas BMI: {lower:.2f} - {upper:.2f}")
    return df

def split_features_target(df):
    """Memisahkan fitur dan target."""
    X = df.drop('Diabetes_binary', axis=1)
    y = df['Diabetes_binary']
    print(f"Fitur: {X.shape[1]} kolom")
    print(f"Target: {y.value_counts().to_dict()}")
    return X, y

def standardize_features(X):
    """Standarisasi fitur menggunakan StandardScaler."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    print("Standarisasi fitur selesai")
    return X_scaled, scaler

def split_data(X, y, test_size=0.2, random_state=42):
    """Split data menjadi train dan test set."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    print(f"Training set : {X_train.shape[0]} baris")
    print(f"Test set     : {X_test.shape[0]} baris")
    return X_train, X_test, y_train, y_test

def save_results(X_train, X_test, y_train, y_test, output_dir):
    """Menyimpan hasil preprocessing ke folder output."""
    os.makedirs(output_dir, exist_ok=True)

    train_df = X_train.copy()
    train_df['Diabetes_binary'] = y_train.values
    train_df.to_csv(os.path.join(output_dir, 'diabetes_train.csv'), index=False)

    test_df = X_test.copy()
    test_df['Diabetes_binary'] = y_test.values
    test_df.to_csv(os.path.join(output_dir, 'diabetes_test.csv'), index=False)

    print(f"File disimpan di: {output_dir}")
    print("- diabetes_train.csv")
    print("- diabetes_test.csv")

def preprocess(filepath, output_dir):
    """Fungsi utama yang menjalankan seluruh pipeline preprocessing."""
    print("="*50)
    print("MULAI PREPROCESSING")
    print("="*50)

    df = load_data(filepath)
    df = remove_duplicates(df)
    df = handle_missing_values(df)
    df = remove_outliers(df)
    X, y = split_features_target(df)
    X_scaled, scaler = standardize_features(X)
    X_train, X_test, y_train, y_test = split_data(X_scaled, y)
    save_results(X_train, X_test, y_train, y_test, output_dir)

    print("="*50)
    print("PREPROCESSING SELESAI!")
    print("="*50)
    return X_train, X_test, y_train, y_test, scaler

if __name__ == "__main__":
    filepath = "../diabetes_raw/diabetes_binary_health_indicators_BRFSS2015.csv"
    output_dir = "../preprocessing"
    preprocess(filepath, output_dir)