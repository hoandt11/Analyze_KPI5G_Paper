import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset



# 1. Đọc file CSV và chọn cột
file_path = r'D:\Data_Paper_5G\DATA\TTML.csv'
df = pd.read_csv(file_path)

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
df = df[selected_columns]

# Chuyển DATETIME_ID thành datetime (giả sử format YYYYMMDDHH)
df['DATETIME_ID'] = pd.to_datetime(df['DATETIME_ID'].astype(str), format='%Y%m%d%H')

# Sort theo thời gian và TTML
df = df.sort_values(by=['TTML', 'DATETIME_ID']).reset_index(drop=True)

# Xử lý missing values (nếu có, điền mean của cột)
df.fillna(df.mean(numeric_only=True), inplace=True)

# 2. Phân tích dữ liệu
print("Thông tin DataFrame:")
print(df.info())

print("\nThống kê mô tả:")
print(df.describe())

print("\nCorrelation matrix (chỉ cột numeric):")
corr = df.corr(numeric_only=True)
print(corr)

# Kiểm tra missing values
print("\nMissing values:")
print(df.isnull().sum())

# 3. Vẽ báo cáo (biểu đồ line cho các KPI chính theo thời gian, group by TTML)
kpi_columns = ['PS_CSSR_NR', 'SDR_NR', 'PRB_UTIL_DL_NR', 'LATENCY_NR', 'USER_DL_THP_NR', 'USER_UL_THP_NR']  # Chọn vài KPI chính để vẽ (bạn có thể thêm)

for kpi in kpi_columns:
    plt.figure(figsize=(12, 6))
    for ttml in df['TTML'].unique():
        subset = df[df['TTML'] == ttml]
        plt.plot(subset['DATETIME_ID'], subset[kpi], label=ttml)
    plt.title(f'Biểu đồ {kpi} theo thời gian')
    plt.xlabel('Thời gian')
    plt.ylabel(kpi)
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{kpi}_report.png')  # Lưu file ảnh
    plt.show()

# Vẽ heatmap correlation
plt.figure(figsize=(16, 12))
mask = np.triu(np.ones_like(corr, dtype=bool))  

sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)

plt.title('Ma trận tương quan các chỉ số KPI', fontsize=18, pad=20)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# 4. Mô hình LSTM để dự đoán KPI (ví dụ: USER_DL_THP_NR)
# Chuẩn bị data: Group by TTML, chọn target = 'USER_DL_THP_NR'
target = 'USER_DL_THP_NR' # Thay bằng KPI bạn muốn dự đoán
sequence_length = 10 # Độ dài sequence (window size)
n_steps = 5 # Dự đoán bao nhiêu bước tiếp theo
# Hàm tạo sequences cho LSTM
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)
# Model LSTM đơn giản
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
   
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out
# Dự đoán cho từng TTML
for ttml in df['TTML'].unique():
    print(f"\nDự đoán cho {ttml}:")
   
    subset = df[df['TTML'] == ttml][[target]].values # Chỉ lấy target column
   
    if len(subset) < sequence_length + 1:
        print(f"Dữ liệu cho {ttml} quá ít ({len(subset)} dòng), bỏ qua LSTM.")
        continue
   
    # Scale data
    scaler = MinMaxScaler()
    subset_scaled = scaler.fit_transform(subset)
   
    # Tạo sequences
    X, y = create_sequences(subset_scaled, sequence_length)
    X = X.reshape((X.shape[0], X.shape[1], 1)) # Reshape cho LSTM [samples, timesteps, features]
   
    # Chia train/test (80/20)
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
   
    # DataLoader
    train_data = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
   
    # Khởi tạo model
    model = LSTMModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
   
    # Train
    epochs = 50 # Điều chỉnh nếu cần
    for epoch in range(epochs):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
   
    # Dự đoán trên test
    model.eval()
    with torch.no_grad():
        predicted = model(torch.from_numpy(X_test).float()).numpy()
    predicted = scaler.inverse_transform(predicted)
    y_test_original = scaler.inverse_transform(y_test)  # Đổi tên để rõ ràng
   
    # Tính metrics đánh giá
    mse = mean_squared_error(y_test_original, predicted)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_original, predicted)
    r2 = r2_score(y_test_original, predicted)
   
    print(f"Đánh giá trên test set cho {ttml}:")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R² Score: {r2:.4f}")
   
    print(f"Dự đoán trên test set cho {ttml}:")
    print(predicted.flatten())
   
    # Dự đoán tương lai (n_steps bước)
    last_sequence = subset_scaled[-sequence_length:].reshape(1, sequence_length, 1)
    future_preds = []
    for _ in range(n_steps):
        with torch.no_grad():
            pred = model(torch.from_numpy(last_sequence).float()).numpy()
        future_preds.append(pred[0][0])
        last_sequence = np.append(last_sequence[:, 1:, :], pred.reshape(1, 1, 1), axis=1)
   
    future_preds = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1))
    print(f"Dự đoán {n_steps} bước tiếp theo cho {ttml}:")
    print(future_preds.flatten())
   
    # Vẽ dự đoán
    plt.figure(figsize=(12, 6))
    plt.plot(subset, label='Thực tế')
    test_index = np.arange(len(subset) - len(y_test), len(subset))
    plt.plot(test_index, predicted, label='Dự đoán test')
    future_index = np.arange(len(subset), len(subset) + n_steps)
    plt.plot(future_index, future_preds, label='Dự đoán tương lai')
    plt.title(f'Dự đoán {target} cho {ttml}')
    plt.legend()
    plt.savefig(f'lstm_{target}_{ttml}.png')
    plt.show()