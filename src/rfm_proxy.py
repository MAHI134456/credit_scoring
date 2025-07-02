import pandas as pd
from datetime import datetime
from datetime import datetime
import pytz
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# calculate RFM (Recency, Frequency, Monetary) values for customers
# Load your preprocessed data
df = pd.read_csv('data/processed/model_ready.csv')  # or your raw transactions file

# Use the original transactions for RFM (not OHE version!)
transactions = pd.read_csv('data/processed/credit_data_clean.csv')

# Use UTC, for example
snapshot_date = datetime(2025, 6, 30, tzinfo=pytz.UTC)

# Make sure TransactionStartTime is datetime
transactions['TransactionStartTime'] = pd.to_datetime(transactions['TransactionStartTime'])

# Calculate RFM
rfm = transactions.groupby('CustomerId').agg({
    'TransactionStartTime': lambda x: (snapshot_date - x.max()).days,
    'TransactionId': 'count',
    'Amount': 'sum'
}).reset_index()

rfm.columns = ['CustomerId', 'Recency', 'Frequency', 'Monetary']

print(rfm.head())


# scale the RFM values
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])
print("Scaled RFM values:")
print(rfm_scaled[:5])

#cluster customers using KMeans


kmeans = KMeans(n_clusters=3, random_state=42)
rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)
print("Cluster assignments:")
print(rfm[['CustomerId', 'Cluster']].head())

#Identify the “High-Risk” Cluster
cluster_profile = rfm.groupby('Cluster').agg({
    'Recency': 'mean',
    'Frequency': 'mean',
    'Monetary': 'mean'
}).reset_index()

print(cluster_profile)

# Create is_high_risk
# Suppose Cluster 2 is the worst based on your printout:
high_risk_cluster = cluster_profile.sort_values(['Recency', 'Frequency', 'Monetary'], ascending=[False, True, True]).iloc[0]['Cluster']

rfm['is_high_risk'] = rfm['Cluster'].apply(lambda x: 1 if x == high_risk_cluster else 0)
rfm = rfm[['CustomerId', 'is_high_risk']]



# Always load from file so you have fresh copy
final_df = pd.read_csv('data/processed/model_ready.csv')

# DEBUG: See columns
print("Before strip - final_df:", final_df.columns)
print("Before strip - rfm:", rfm.columns)

# Strip spaces on final_df and rfm ONLY
final_df.columns = final_df.columns.str.strip()
rfm.columns = rfm.columns.str.strip()

print("After strip - final_df:", final_df.columns)
print("After strip - rfm:", rfm.columns)

# ✅ Safe test
assert 'CustomerId' in final_df.columns, "CustomerId missing in final_df!"
assert 'CustomerId' in rfm.columns, "CustomerId missing in rfm!"

# Merge on CustomerId
final_df = pd.merge(final_df, rfm, on='CustomerId', how='left')

# Fill any missing labels
final_df['is_high_risk'] = final_df['is_high_risk'].fillna(0).astype(int)

print(final_df['is_high_risk'].value_counts())


# Save
final_df.to_csv('data/processed/model_ready_with_target.csv', index=False)

print("✅ Saved with proxy target: data/processed/model_ready_with_target.csv")
