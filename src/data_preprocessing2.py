# src/data_processing.py

import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# If you want to use WOE later, uncomment this
# from xverse.transformer import WOE


# =============================
# Custom Transformer for Aggregates
# =============================

class AggregateFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, group_col='CustomerId', target_col='Amount'):
        self.group_col = group_col
        self.target_col = target_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        agg_df = (
            X.groupby(self.group_col)[self.target_col]
            .agg(['sum', 'mean', 'count', 'std'])
            .reset_index()
            .rename(columns={
                'sum': f'{self.target_col}_sum',
                'mean': f'{self.target_col}_mean',
                'count': f'{self.target_col}_count',
                'std': f'{self.target_col}_std'
            })
        )
        X = X.merge(agg_df, on=self.group_col, how='left')
        return X


# =============================
# Custom Transformer for Datetime
# =============================

class DatetimeFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, datetime_col='TransactionStartTime'):
        self.datetime_col = datetime_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X[self.datetime_col] = pd.to_datetime(X[self.datetime_col], errors='coerce')
        X['transaction_hour'] = X[self.datetime_col].dt.hour
        X['transaction_day'] = X[self.datetime_col].dt.day
        X['transaction_month'] = X[self.datetime_col].dt.month
        X['transaction_year'] = X[self.datetime_col].dt.year
        return X


# =============================
# Main build_pipeline() function
# =============================

def build_pipeline(df):
    """
    Builds a sklearn Pipeline for full feature engineering.
    """

    # ID columns — use for grouping only, not for encoding
    id_cols = ['TransactionId', 'BatchId', 'AccountId', 'SubscriptionId', 'CustomerId']
    datetime_col = 'TransactionStartTime'

    # Get numeric features: exclude IDs!
    num_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    num_features = [col for col in num_features if col not in id_cols]

    # Final list of categorical features — exclude IDs & datetime!
    cat_features = [
        'CurrencyCode',
        'ProviderId',
        'ProductId',
        'ProductCategory',
        'ChannelId'
    ]

    print("✅ Numeric features:", num_features)
    print("✅ Categorical features:", cat_features)

    # Steps for numeric columns
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Steps for categorical columns
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Combine all pipelines
    preprocessor = ColumnTransformer([
        ('num', num_pipeline, num_features),
        ('cat', cat_pipeline, cat_features)
    ])

    # Final pipeline with custom feature steps
    full_pipeline = Pipeline([
        ('datetime', DatetimeFeatures(datetime_col=datetime_col)),
        ('aggregate', AggregateFeatures(group_col='CustomerId', target_col='Amount')),
        ('preprocessor', preprocessor)
    ])

    return full_pipeline


# =============================
# Example usage: run directly
# =============================

if __name__ == "__main__":
    df = pd.read_csv('data/processed/credit_data_clean.csv')

    pipeline = build_pipeline(df)

    X_transformed = pipeline.fit_transform(df)

    print(f"✅ Pipeline done. Shape: {X_transformed.shape}")

    # Save to CSV if you want
    X_transformed_df = pd.DataFrame(X_transformed)
    X_transformed_df.to_csv('data/processed/model_ready.csv', index=False)

    print("✅ Saved to data/processed/model_ready.csv")

    # Load your cleaned data
df = pd.read_csv('data/processed/credit_data_clean.csv')

# ✅ Save CustomerId separately
ids = df[['CustomerId']].reset_index(drop=True)

# ✅ Build your pipeline as usual
pipeline = build_pipeline(df)

# ✅ Transform your features
X_transformed = pipeline.fit_transform(df)

# ✅ Convert to DataFrame
X_transformed_df = pd.DataFrame(X_transformed).reset_index(drop=True)

# ✅ Concatenate IDs back
final_df = pd.concat([ids, X_transformed_df], axis=1)

# ✅ Save model-ready data WITH CustomerId
final_df.to_csv('data/processed/model_ready.csv', index=False)

print("✅ model_ready.csv now includes CustomerId")
print(final_df.columns)
