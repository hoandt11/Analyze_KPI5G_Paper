import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# ================================
# 1. READ DATA
# ================================
file_path = r'D:\Data_Paper_5G\DATA\TTML.csv'
df = pd.read_csv(file_path)

# Columns to keep (exact names from the CSV)
selected_columns = [
    'DATETIME_ID',
    'TTML',
    'PS_CSSR_NR',
    'SDR_NR',
    'PRB_UTIL_DL_NR',
    'PRB_UTIL_UL_NR',
    'LATENCY_NR',
    'PKTLOSSR',
    'CONNECTED_RRC_USER_AVERAGE',
    'CONNECTED_RRC_USER_MAX',
    'DKD5G_NR',
    'EN_DC_SR_NR',
    'HOSR_NR',
    'RASR_NR',
    'DL_TRAFFIC_NR',
    'UL_TRAFFIC_NR',
    'USER_DL_THP_NR',
    'USER_UL_THP_NR',
    'CELL_DL_THP_NR',
    'CELL_UL_THP_NR'
]

# Keep only selected columns
df = df[selected_columns]

# Drop rows with any missing value in the selected columns
df = df.dropna(subset=selected_columns)
df = df.reset_index(drop=True)

# -------------------------------------------------
# YOUR REQUESTED VIETNAMESE PRINT BLOCK
# -------------------------------------------------
print(f"Đã làm sạch dữ liệu: {df.shape[0]} dòng, {df.shape[1]} cột")
print("\n5 dòng đầu:")
print(df.head())
# -------------------------------------------------

# -------------------------------------------------
# BASIC INFORMATION BLOCK (English)
# -------------------------------------------------
print("\nData information:")
print(df.info())

print("\nStatistical description:")
print(df.describe())

print("\nFirst few rows:")
print(df.head())

# Check missing values (should be 0 after dropna)
missing = df.isnull().sum()
print("\nMissing values (should be 0):")
print(missing[missing > 0])
# -------------------------------------------------

# ================================
# 2. BASIC PRE-PROCESSING
# ================================
df['DATETIME_ID'] = pd.to_datetime(df['DATETIME_ID'], format='%Y%m%d%H')
df['DATE'] = df['DATETIME_ID'].dt.date
df['HOUR'] = df['DATETIME_ID'].dt.hour

# ================================
# 3. EXPLORATORY VISUALISATIONS
# ================================

# Plot 1: USER_DL_THP_NR by region over time
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x='DATETIME_ID', y='USER_DL_THP_NR', hue='TTML', marker='o')
plt.title('User Downlink Throughput (Mbps) by Region and Time')
plt.xticks(rotation=45)
plt.ylabel('USER_DL_THP_NR (Mbps)')
plt.legend(title='Region')
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot 2: Correlation heatmap
numeric_cols = df.select_dtypes(include=[np.number]).columns
corr = df[numeric_cols].corr()
plt.figure(figsize=(14, 10))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix of KPIs')
plt.tight_layout()
plt.show()

# Plot 3: Distribution
plt.figure(figsize=(10, 5))
sns.histplot(df['USER_DL_THP_NR'], kde=True, bins=30)
plt.title('Distribution of USER_DL_THP_NR')
plt.xlabel('Mbps')
plt.show()

# ================================
# 4. RANDOM FOREST PREDICTION
# ================================

target = 'USER_DL_THP_NR'

feature_cols = [
    'PS_CSSR_NR', 'SDR_NR', 'PRB_UTIL_DL_NR', 'PRB_UTIL_UL_NR',
    'LATENCY_NR', 'PKTLOSSR', 'CONNECTED_RRC_USER_AVERAGE',
    'CONNECTED_RRC_USER_MAX', 'DKD5G_NR', 'EN_DC_SR_NR',
    'HOSR_NR', 'RASR_NR', 'DL_TRAFFIC_NR', 'UL_TRAFFIC_NR',
    'CELL_DL_THP_NR', 'CELL_UL_THP_NR'   # No extra spaces!
]

df_encoded = pd.get_dummies(df, columns=['TTML'], prefix='REGION')
feature_cols += [c for c in df_encoded.columns if c.startswith('REGION_')]

X = df_encoded[feature_cols]
y = df_encoded[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("\n=== RANDOM FOREST RESULTS ===")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.4f}")
print(f"R² Score: {r2_score(y_test, y_pred):.4f}")

# Feature importance
importances = pd.Series(model.feature_importances_, index=feature_cols)
top10 = importances.sort_values(ascending=False).head(10)
plt.figure(figsize=(10, 6))
top10.plot(kind='barh')
plt.title('Top 10 Most Important Features')
plt.xlabel('Importance')
plt.tight_layout()
plt.show()

# ================================
# 5. PREDICT NEW RECORD
# ================================
new_record = {
    'PS_CSSR_NR': 99.5, 'SDR_NR': 0.1, 'PRB_UTIL_DL_NR': 1.5, 'PRB_UTIL_UL_NR': 1.8,
    'LATENCY_NR': 20000, 'PKTLOSSR': 0.0, 'CONNECTED_RRC_USER_AVERAGE': 5000,
    'CONNECTED_RRC_USER_MAX': 20000, 'DKD5G_NR': 98.5, 'EN_DC_SR_NR': 99.8,
    'HOSR_NR': 95.0, 'RASR_NR': 98.0, 'DL_TRAFFIC_NR': 2.5e7, 'UL_TRAFFIC_NR': 2.0e6,
    'CELL_DL_THP_NR': 300, 'CELL_UL_THP_NR': 15,
    'REGION_Mien Bac': 1, 'REGION_Mien Nam': 0, 'REGION_Mien Trung': 0, 'REGION_Mobifone': 0
}

pred = model.predict(pd.DataFrame([new_record]))[0]
print(f"\nPREDICTION: USER_DL_THP_NR ≈ {pred:.2f} Mbps")