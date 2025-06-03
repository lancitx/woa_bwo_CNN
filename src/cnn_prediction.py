import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler

# 支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 原始数据
data = np.array([
    [5.33, 5.39, 5.29, 5.41, 5.45, 5.50],
    [5.39, 5.29, 5.41, 5.50, 5.57, 5.57],
    [5.29, 5.41, 5.45, 5.50, 5.57, 5.58],
    [5.41, 5.45, 5.50, 5.57, 5.58, 5.61],
    [5.45, 5.50, 5.57, 5.58, 5.61, 5.69],
    [5.50, 5.57, 5.58, 5.61, 5.69, 5.78],
    [5.57, 5.58, 5.61, 5.69, 5.78, 5.78],
    [5.58, 5.61, 5.69, 5.78, 5.78, 5.81],
    [5.61, 5.69, 5.78, 5.78, 5.81, 5.86],
    [5.69, 5.78, 5.78, 5.81, 5.86, 5.90],
    [5.78, 5.78, 5.81, 5.86, 5.90, 5.97],
    [5.78, 5.81, 5.86, 5.90, 5.97, 6.49],
    [5.81, 5.86, 5.90, 5.97, 6.49, 6.60],
    [5.86, 5.90, 5.97, 6.49, 6.60, 6.64],
    [5.90, 5.97, 6.49, 6.60, 6.64, 6.74],
    [5.97, 6.49, 6.60, 6.64, 6.74, 6.87],
    [6.49, 6.60, 6.64, 6.74, 7.87, 7.01]
])

# 分离输入和输出
X = data[:, :5]  # 前5列作为输入特征
y = data[:, 5]   # 最后一列作为输出目标

# 划分数据集
X_train = X[:14]  # 前14组训练
y_train = y[:14]  # 前14组目标值

X_test = X[14:]   # 后3组测试
y_test_true = y[14:]  # 测试集真实值（最后一行NaN）

# 数据标准化
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()

# 重塑数据为3D格式 [样本数, 时间步长, 特征数]
X_train_3d = X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
X_test_3d = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))

# 创建修正后的CNN模型
model = Sequential()
# 修改1：使用较小的卷积核并移除池化层
model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(5, 1), padding='same'))
# 修改2：使用全局池化替代展平层
model.add(Conv1D(filters=32, kernel_size=2, activation='relu', padding='same'))
model.add(Conv1D(filters=16, kernel_size=2, activation='relu', padding='same'))
model.add(GlobalMaxPooling1D())  # 全局最大池化
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1))

# 编译模型
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='mse',
              metrics=['mae'])

# 打印模型摘要
model.summary()

# 训练模型
history = model.fit(X_train_3d, y_train_scaled,
                    epochs=300,
                    batch_size=4,
                    verbose=1,
                    validation_split=0.2)

# 预测测试集
y_test_scaled = model.predict(X_test_3d)
y_test_pred = scaler_y.inverse_transform(y_test_scaled).flatten()

# 打印预测结果
print("\n测试集预测结果：")
for i, (true, pred) in enumerate(zip(y_test_true, y_test_pred)):
    if not np.isnan(true):  # 跳过缺失值
        print(f"样本 {i+15}: 真实值 = {true:.4f}, 预测值 = {pred:.4f}")
    else:
        print(f"样本 {i+15}: 真实值 = 缺失, 预测值 = {pred:.4f}")

# 评估模型（只评估有真实值的样本）
valid_indices = ~np.isnan(y_test_true)
if np.sum(valid_indices) > 0:
    test_mse = np.mean((y_test_pred[valid_indices] - y_test_true[valid_indices])**2)
    test_mae = np.mean(np.abs(y_test_pred[valid_indices] - y_test_true[valid_indices]))
    print(f"\n模型评估 (测试样本):")
    print(f"均方误差 (MSE): {test_mse:.6f}")
    print(f"平均绝对误差 (MAE): {test_mae:.6f}")

# 预测缺失的样本
print("\n样本17预测值:", f"{y_test_pred[2]:.4f} (10^4 km)")

# === 绘制损失曲线 ===
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='训练损失', linewidth=2, marker='o')
plt.plot(history.history['val_loss'], label='验证损失', linewidth=2, linestyle='--', marker='s')
plt.title('损失曲线')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()

# === 绘制预测结果 ===
plt.subplot(1, 2, 2)
plt.plot(range(1, 15), y_train, 'bo-', label='训练集实际值')
for i in range(3):
    idx = 14 + i + 1
    plt.scatter(idx, y_test_true[i], color='g', marker='^', s=80, label='测试集实际值' if i == 0 else '')
    plt.scatter(idx, y_test_pred[i], color='r', marker='s', s=80, label='测试集预测值' if i == 0 else '')
plt.plot(range(1, 18), np.concatenate([y_train, y_test_pred]), 'r--', alpha=0.6, label='预测趋势')
plt.title('预测结果比较')
plt.xlabel('样本编号')
plt.ylabel('里程 (10^4 km)')
plt.xticks(range(1, 18))
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig('..\img\cnn_prediction_final.png', dpi=100)
plt.show()

print("\n图表已生成并保存为 'cnn_prediction_final.png'")