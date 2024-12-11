import pandas as pd
import numpy as np
import re
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin


class DropDuplicates(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop_duplicates()

class DropDuplicateDescriptions(BaseEstimator, TransformerMixin):
    def __init__(self, target_column):
        self.target_column = target_column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features = X.drop(columns=[self.target_column], errors='ignore')
        duplicate_indices = features[features.duplicated(keep='first')].index
        X = X.drop('selling_price', axis = 1)
        return X.drop(index=duplicate_indices, errors='ignore').reset_index(drop=True)

class ExtractNumericValues(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        for column in self.columns:
            if column in X_copy.columns:
                X_copy[column] = X_copy[column].astype(str).str.extract(r'(\d+\.?\d*)', expand=False).astype(float)
            else:
                print(f"Warning: Column '{column}' not found in DataFrame.")
        return X_copy

def process_torque(row):
    if not isinstance(row, str) or pd.isna(row):
        return np.nan, np.nan

    try:
        clean_row = row.replace(",", "").replace("(", "").replace(")", "").replace("~", "-").strip().lower()
        numbers = [float(num) for num in re.findall(r"[\d.]+", clean_row)]

        if not numbers:
            return np.nan, np.nan

        if 'kgm' in clean_row:
            torque_value = min(numbers) * 9.8
            max_rpm = max(numbers)
        elif 'nm' in clean_row:
            torque_value = min(numbers)
            max_rpm = max(numbers)
        else:
            return np.nan, np.nan

        return round(torque_value, 2), max_rpm

    except Exception as e:
        print(f"Error processing row: {row} -> {e}")
        return np.nan, np.nan

class TorqueProcessor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X[['torque', 'max_torque_rpm']] = X['torque'].apply(lambda x: pd.Series(process_torque(x)))
        return X

class ConvertToInt(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=numerical_features + [col for col in X_train_new.drop('selling_price',axis = 1).columns if col not in numerical_features])
        X_copy = X.copy()
        for column in self.columns:
            if column in X_copy.columns:
                X_copy[column] = pd.to_numeric(X_copy[column], errors='coerce')
                X_copy[column] = X_copy[column].fillna(0).astype(int)
            else:
                print(f"Warning: Column '{column}' not found in DataFrame.")
        return X_copy

class ProcessName(BaseEstimator, TransformerMixin):
    def __init__(self, stop_words, min_count=2):
        self.stop_words = stop_words
        self.min_count = min_count
        self.name_counts = []
        self.frequent_names = None

    def fit(self, X, y=None):
        self.name_counts = X['name'].value_counts()
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy['name'] = X_copy['name'].str.lower()
        X_copy['name'] = X_copy['name'].str.strip()
        X_copy['name'] = X_copy['name'].str.replace(r'\s+', ' ', regex=True)
        X_copy['name'] = X_copy['name'].apply(lambda x: ' '.join([word for word in x.split() if word not in self.stop_words]))
        X_copy['name'] = X_copy['name'].apply(lambda x: ' '.join(x.split()[:2]))

        self.name_counts = X_copy['name'].value_counts()
        X_copy['name'] = X_copy['name'].apply(lambda x: x if self.name_counts[x] >= 5 else 'other')
        return X_copy

class ResetIndex(BaseEstimator, TransformerMixin):
    def __init__(self, drop=True):
        self.drop = drop

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.reset_index(drop=self.drop)

class ReplaceOutliersAndZeros(BaseEstimator, TransformerMixin):
    def __init__(self, torque_threshold=1000, rpm_threshold=10000):
        self.torque_threshold = torque_threshold
        self.rpm_threshold = rpm_threshold

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy.loc[X_copy['torque'] > self.torque_threshold, 'torque'] = X_copy['torque'].median()
        X_copy.loc[X_copy['max_torque_rpm'] > self.rpm_threshold, 'max_torque_rpm'] = X_copy['max_torque_rpm'].median()
        X_copy.loc[X_copy['max_power'] == 0, 'max_power'] = X_copy['max_power'].median()
        return X_copy


class AddPowerPerEngine(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X['power_per_engine'] = X['max_power'] / X['engine']
        return X

class FillNaWithZeros(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.fillna(0)


class CustomEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        self.encoder = OneHotEncoder(drop="first", sparse_output=False, handle_unknown='ignore')
        self.encoded_columns = None

    def fit(self, X, y=None):
        self.encoder.fit(X[self.columns])
        self.encoded_columns = self.encoder.get_feature_names_out(self.columns)
        return self

    def transform(self, X):
        encoded_data = self.encoder.transform(X[self.columns])
        encoded_df = pd.DataFrame(encoded_data, columns=self.encoded_columns, index=X.index)
        X_transformed = pd.concat([X.drop(columns=self.columns, errors='ignore'), encoded_df], axis=1)
        return X_transformed

class CustomScaler(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        self.scaler.fit(X[self.columns])
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy[self.columns] = self.scaler.transform(X_copy[self.columns])
        return X_copy

numerical_features = ['year', 'engine', 'mileage', 'max_power', 'torque', 'max_torque_rpm', 'seats']
categorical_columns = ['name', 'fuel', 'seller_type', 'transmission', 'owner', 'seats']
numeric_columns = ['year', 'km_driven', 'mileage', 'engine', 'max_power', 'torque', 'max_torque_rpm', 'power_per_engine']
stop_words = ['vdi', 'bsiii', 'tdi', 'vtvt', 'diesel', 'petrol', 'electric', 'ambition', 'sportz']



df_train = pd.read_csv('https://raw.githubusercontent.com/Murcha1990/MLDS_ML_2022/main/Hometasks/HT1/cars_train.csv')
df_test = pd.read_csv('https://raw.githubusercontent.com/Murcha1990/MLDS_ML_2022/main/Hometasks/HT1/cars_test.csv')

X_train_new = df_train
y_train = df_train['selling_price']
X_test_new = df_test
y_test = df_test['selling_price']