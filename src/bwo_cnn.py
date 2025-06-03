import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm   # 用于显示进度条
import matplotlib.pyplot as plt
import matplotlib

# —— 配置 matplotlib 中文显示 ——
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# -----------------------------
# 1. 数据准备与预处理
# -----------------------------
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

# 输入特征：前5 列；目标：第6 列
X = data[:, :5]
y = data[:, 5]

# 前14 条做训练，后 3 条做测试
X_train = X[:14, :]
y_train = y[:14]
X_test  = X[14:, :]
y_test_true = y[14:]   # 真实值

# 数据标准化
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled  = scaler_X.transform(X_test)

scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()

# 重塑为 CNN 输入所需 3D 形状：[样本数, 时间步长, 特征数]
X_train_3d = X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
X_test_3d  = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))


# -----------------------------
# 2. 定义 CNN 构建函数
# -----------------------------
def build_cnn_model(hyperparams):
    """
    根据超参数向量构建并返回编译好的 CNN 模型。
    hyperparams: 长度为 4 的可迭代对象，对应 [filters1, filters2, dense_units, lr]
    """
    # 解析并转换超参数
    f1 = int(round(hyperparams[0]))      # Conv1D 第一层滤波器个数
    f2 = int(round(hyperparams[1]))      # Conv1D 第二层滤波器个数
    fc = int(round(hyperparams[2]))      # 全连接层神经元个数
    lr = float(hyperparams[3])           # 学习率

    model = Sequential([
        Conv1D(filters=f1, kernel_size=2, activation='relu', padding='same', input_shape=(5, 1)),
        Conv1D(filters=f2, kernel_size=2, activation='relu', padding='same'),
        GlobalMaxPooling1D(),
        Dense(fc, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])
    optimizer = Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    return model


# -----------------------------
# 3. 定义适应度函数
# -----------------------------
def fitness_function(hyperparams):
    """
    给定超参数向量 hyperparams，构建 CNN 并短轮次训练，
    返回验证集上的最低 MSE 作为适应度值（目标是最小化该值）。
    hyperparams: np.array([f1, f2, fc, lr])
    """
    # 限制超参数在合理区间，否则返回一个很大的惩罚值
    f1, f2, fc, lr = hyperparams
    if not (16 <= f1 <= 128 and 8 <= f2 <= 64 and 16 <= fc <= 128 and 1e-4 <= lr <= 1e-2):
        return 1e6

    # 构建模型并训练少量 epoch
    model = build_cnn_model(hyperparams)
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(
        X_train_3d, y_train_scaled,
        epochs=30,
        batch_size=4,
        verbose=0,
        validation_split=0.2,
        callbacks=[early_stop]
    )

    # 返回验证集上最小 val_loss（MSE）
    val_mse = min(history.history['val_loss'])
    return val_mse


# -----------------------------
# 4. 实现 BWO（Beluga Whale Optimization）算法
# -----------------------------
def beluga_whale_optimization(n_whales=10, max_iter=20):
    """
    Beluga Whale Optimization (BWO)：
    用于在 4 维超参数空间中搜索最优值。
    n_whales: 鲸鱼数量
    max_iter: 最大迭代次数
    返回：(最优超参数向量, 最优适应度)
    """
    # 每个维度的边界：[f1_min, f1_max], [f2_min, f2_max], [fc_min, fc_max], [lr_min, lr_max]
    lb = np.array([16, 8, 16, 1e-4], dtype=float)
    ub = np.array([128, 64, 128, 1e-2], dtype=float)
    dim = 4  # 4 维超参数

    # 1. 随机初始化 n_whales 条 Beluga Whale 的“位置”向量
    X = np.random.uniform(low=lb, high=ub, size=(n_whales, dim))
    fitness = np.full(n_whales, np.inf)

    # 先评估初始鲸鱼的适应度
    for i in range(n_whales):
        fitness[i] = fitness_function(X[i])

    # 记录全局最优
    best_idx = np.argmin(fitness)
    X_best = X[best_idx].copy()
    best_fit = fitness[best_idx]

    # 2. 迭代更新位置
    for t in tqdm(range(max_iter), desc="BWO 优化进度"):
        # Beluga Whale 特有参数：让搜索更加“探索-开发”自适应
        # 这里我们用 r1, r2 控制收敛率，用 b 控制螺旋系数
        a = 2 - 2 * (t / max_iter)   # a 从 2 线性减小到 0

        for i in range(n_whales):
            r1 = random.random()
            r2 = random.random()
            A = 2 * a * r1 - a         # A 控制“包围”或“探索”
            C = 2 * r2                 # C 控制搜索范围
            b = 1                      # 常数，螺旋更新时用到
            l = random.uniform(-1, 1)  # 用于螺旋公式中的随机项

            p = random.random()
            if p < 0.5:
                # —— Beluga“包围猎物”/“随机搜索”机制 ——
                if abs(A) < 1:
                    # 靠近当前全局最优鲸鱼 X_best
                    D = np.abs(C * X_best - X[i])
                    X_new = X_best - A * D
                else:
                    # 选一只随机鲸鱼 X_rand，进行探索
                    rand_idx = random.randint(0, n_whales - 1)
                    X_rand = X[rand_idx]
                    D = np.abs(C * X_rand - X[i])
                    X_new = X_rand - A * D
            else:
                # —— Beluga“螺旋更新”机制 ——
                D_best = np.abs(X_best - X[i])
                # 这里的 BWO 螺旋：在 WOA 基础上增加一个随机系数 b
                X_new = D_best * np.exp(b * l) * np.cos(2 * np.pi * l) + X_best

            # 新位置做边界限制
            X_new = np.clip(X_new, lb, ub)

            # 评估新位置的适应度
            f_new = fitness_function(X_new)
            if f_new < fitness[i]:
                X[i] = X_new.copy()
                fitness[i] = f_new

            # 更新全局最优
            if f_new < best_fit:
                X_best = X_new.copy()
                best_fit = f_new

        # （可选）每代结束时打印最优适应度
        # print(f"第 {t+1} 代：最优验证 MSE = {best_fit:.6f}")

    return X_best, best_fit


# -----------------------------
# 5. 主流程：调用 BWO，训练最终模型，预测 & 可视化
# -----------------------------
if __name__ == "__main__":
    # 为了结果可重复，固定随机种子
    np.random.seed(42)
    tf.random.set_seed(42)
    random.seed(42)

    print("开始 BWO 寻优 ...")
    best_params, best_val_mse = beluga_whale_optimization(n_whales=10, max_iter=30)
    print(f"\nBWO 最优超参数 (f1, f2, fc, lr) = {best_params}")
    print(f"对应验证集最小 MSE = {best_val_mse:.6f}")

    # 将浮点超参数转换为实际整数
    f1_opt = int(round(best_params[0]))
    f2_opt = int(round(best_params[1]))
    fc_opt = int(round(best_params[2]))
    lr_opt = float(best_params[3])
    print(f"\n实际采用的超参数：filters1={f1_opt}, filters2={f2_opt}, dense_units={fc_opt}, lr={lr_opt:.5f}\n")

    # 用找到的超参数重新训练“最终模型”（完整训练）
    model_opt = Sequential([
        Conv1D(filters=f1_opt, kernel_size=2, activation='relu', padding='same', input_shape=(5, 1)),
        Conv1D(filters=f2_opt, kernel_size=2, activation='relu', padding='same'),
        GlobalMaxPooling1D(),
        Dense(fc_opt, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])
    optimizer_opt = Adam(learning_rate=lr_opt)
    model_opt.compile(optimizer=optimizer_opt, loss='mse', metrics=['mae'])

    print("开始用最优超参数训练最终模型 ...")
    early_stop_final = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history_opt = model_opt.fit(
        X_train_3d, y_train_scaled,
        epochs=100,
        batch_size=4,
        validation_split=0.2,
        callbacks=[early_stop_final],
        verbose=1
    )

    # 在测试集上预测
    y_test_pred_scaled = model_opt.predict(X_test_3d)
    y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled).flatten()

    print("\n最终模型在测试集上的预测结果：")
    for i, (true_val, pred_val) in enumerate(zip(y_test_true, y_test_pred)):
        print(f"样本{i+15}：真实值={true_val:.4f}，预测值={pred_val:.4f}，误差={(pred_val - true_val):.4f}")

    # 可视化 1：训练 & 验证损失曲线
    plt.figure(figsize=(10, 4), dpi=100)
    plt.subplot(1, 2, 1)
    plt.plot(history_opt.history['loss'], label='训练损失', linewidth=2, marker='o', markersize=4)
    plt.plot(history_opt.history['val_loss'], label='验证损失', linewidth=2, linestyle='--', marker='s', markersize=4)
    plt.title('最终模型损失曲线', fontsize=14)
    plt.xlabel('Epoch', fontsize=12); plt.ylabel('MSE', fontsize=12)
    plt.xticks(fontsize=10); plt.yticks(fontsize=10)
    plt.grid(linestyle='--', alpha=0.5)
    plt.legend(fontsize=10)

    # 可视化 2：训练集实际 & 测试集实际-预测
    plt.subplot(1, 2, 2)
    idxs = range(1, 18)
    # 训练集真实值
    plt.plot(range(1, 15), y_train, 'bo-', label='训练集真实', linewidth=2, markersize=6)
    # 测试集真实值 & 预测值
    for j in range(3):
        xj = 14 + j + 1
        # 真实值
        plt.scatter(xj, y_test_true[j], color='g', marker='^', s=80,
                    label='测试集真实' if j == 0 else "")
        # 预测值
        plt.scatter(xj, y_test_pred[j], color='r', marker='s', s=80,
                    label='测试集预测' if j == 0 else "")

    # 连线表示整体趋势（可选）
    all_pred = np.concatenate([y_train, y_test_pred])
    plt.plot(idxs, all_pred, 'r--', alpha=0.5, label='整体预测趋势')

    plt.title('最终模型预测结果', fontsize=14)
    plt.xlabel('样本编号', fontsize=12); plt.ylabel('里程 (10^4 km)', fontsize=12)
    plt.xticks(idxs, fontsize=10); plt.yticks(fontsize=10)
    plt.grid(linestyle='--', alpha=0.5)

    # 图例去重
    handles, labels = plt.gca().get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    plt.legend(unique.values(), unique.keys(), fontsize=10, loc='upper left', framealpha=0.9)

    plt.tight_layout()
    plt.savefig('../img/bwo_cnn_final.png', dpi=100)
    plt.show()

    print("\n全部流程结束，图表已保存为 'bwo_cnn_final.png'。")