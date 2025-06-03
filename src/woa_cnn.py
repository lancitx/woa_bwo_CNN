import random
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
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

X = data[:, :5]
y = data[:, 5]

# 训练集：前14条；测试集：后3条
X_train = X[:14, :]
y_train = y[:14]
X_test  = X[14:, :]
y_test_true = y[14:]   # 测试集中最后一行理论上都是已知的

# 归一化 / 标准化
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled  = scaler_X.transform(X_test)

scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()

# 重塑为 CNN 输入需要的 3D 格式 [samples, timesteps, features]
X_train_3d = X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
X_test_3d  = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))

# -----------------------------
# 2. 定义 CNN 构建函数
# -----------------------------
def build_cnn_model(hyperparams):
    """
    根据传入的 hyperparams 构建并返回编译后的 CNN 模型。
    hyperparams: 长度为4的可迭代对象，对应 [filters1, filters2, dense_units, lr]
    """
    # 从超参数中解析出各个数值
    f1 = int(round(hyperparams[0]))  # 第一层卷积滤波器个数
    f2 = int(round(hyperparams[1]))  # 第二层卷积滤波器个数
    fc = int(round(hyperparams[2]))  # 全连接层单元数
    lr = float(hyperparams[3])       # 学习率（浮点数）

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
    给定一组超参数 hyperparams，构建 CNN，并在训练集上训练，返回验证集上的 MSE 作为适应度值。
    hyperparams: np.array([f1, f2, fc, lr])
    """
    # 先把超参数限制在合理范围内，否则不给出很差的适应度
    # 例如 f1 ∈ [16, 128], f2 ∈ [8, 64], fc ∈ [16, 128], lr ∈ [1e-4, 1e-2]
    f1, f2, fc, lr = hyperparams
    if not (16 <= f1 <= 128 and 8 <= f2 <= 64 and 16 <= fc <= 128 and 1e-4 <= lr <= 1e-2):
        return 1e6  # 如果超参数越界，就给一个极大损失

    # 构建模型
    model = build_cnn_model(hyperparams)

    # 训练时只跑很少轮数，例如 30 轮
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(
        X_train_3d, y_train_scaled,
        epochs=30,
        batch_size=4,
        verbose=0,
        validation_split=0.2,
        callbacks=[early_stop]
    )

    # 取最后一个 epoch 的 val_loss 作为适应度（或直接用 history.history['val_loss'] 的最小值）
    val_mse = min(history.history['val_loss'])
    # Keras 返回的 loss 就是  MSE
    return val_mse

# -----------------------------
# 4. 实现 WOA（Whale Optimization）算法
# -----------------------------
def whale_optimization(n_whales=10, max_iter=20):
    """
    简化版 WOA：在 4 维超参数空间中搜索最优点。
    n_whales: 鲸鱼数量
    max_iter: 最大迭代次数
    """
    # 4 维的范围： [f1_min, f1_max], [f2_min, f2_max], [fc_min, fc_max], [lr_min, lr_max]
    lb = np.array([16, 8, 16, 1e-4])   # 下界
    ub = np.array([128, 64, 128, 1e-2]) # 上界

    dim = 4  # 超参数维度

    # 1. 随机初始化 n_whales 只鲸鱼的位置
    X = np.random.uniform(low=lb, high=ub, size=(n_whales, dim))
    fitness = np.full(n_whales, np.inf)

    # 先评估一次初始鲸鱼的适应度
    for i in range(n_whales):
        fitness[i] = fitness_function(X[i])

    # 记录全局最优
    best_idx = np.argmin(fitness)
    X_best = X[best_idx].copy()
    best_fit = fitness[best_idx]

    # 2. 迭代更新
    for t in tqdm(range(max_iter), desc="WOA 优化进度"):
        a = 2 - 2 * (t / max_iter)  # a 从 2 线性减到 0

        for i in range(n_whales):
            r1 = random.random()
            r2 = random.random()
            A = 2 * a * r1 - a
            C = 2 * r2
            p = random.random()

            if p < 0.5:
                # 包围猎物或搜索猎物
                if abs(A) < 1:
                    # 靠近最优鲸鱼 X_best
                    D = abs(C * X_best - X[i])
                    X_new = X_best - A * D
                else:
                    # 随机选择另一只鲸鱼 X_rand
                    rand_idx = random.randint(0, n_whales - 1)
                    X_rand = X[rand_idx]
                    D = abs(C * X_rand - X[i])
                    X_new = X_rand - A * D
            else:
                # 螺旋更新公式
                D_best = abs(X_best - X[i])
                b = 1  # 螺旋常数，一般取 1
                l = random.uniform(-1, 1)
                X_new = D_best * np.exp(b * l) * np.cos(2 * np.pi * l) + X_best

            # 越界检查：如果超出边界，就裁剪回 [lb, ub]
            X_new = np.clip(X_new, lb, ub)
            # 重新评估适应度
            f_new = fitness_function(X_new)

            # 如果新位置更好，则更新鲸鱼 i 的位置
            if f_new < fitness[i]:
                X[i] = X_new.copy()
                fitness[i] = f_new

            # 更新全局最优
            if f_new < best_fit:
                X_best = X_new.copy()
                best_fit = f_new

        # 每一代结束，可以打印当前最优适应度
        # print(f"第 {t+1} 代，最优验证 MSE = {best_fit:.6f}")

    return X_best, best_fit

# -----------------------------
# 5. 主流程：调用 WOA，训练最终模型，预测 & 可视化
# -----------------------------
if __name__ == "__main__":
    # 1. 用 WOA 搜索最优超参数
    print("开始 WOA 寻优 ...")
    best_params, best_val_mse = whale_optimization(n_whales=10, max_iter=30)
    print(f"\nWOA 优化完成，最优超参数（f1, f2, fc, lr）= {best_params}")
    print(f"对应验证集最小 MSE = {best_val_mse:.6f}")

    # 将浮点数转换为整数（对滤波器数和全连接单元数）
    f1_opt = int(round(best_params[0]))
    f2_opt = int(round(best_params[1]))
    fc_opt = int(round(best_params[2]))
    lr_opt = float(best_params[3])
    print(f"\n实际采用的超参数：filters1={f1_opt}, filters2={f2_opt}, dense_units={fc_opt}, lr={lr_opt:.5f}")

    # 2. 用找到的超参数重新训练最终模型（可多跑几个 epoch）
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

    print("\n开始用最优超参数训练最终模型 ...")
    early_stop_final = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history_opt = model_opt.fit(
        X_train_3d, y_train_scaled,
        epochs=100,
        batch_size=4,
        validation_split=0.2,
        callbacks=[early_stop_final],
        verbose=1
    )

    # 3. 在测试集上做预测并输出结果
    y_test_pred_scaled = model_opt.predict(X_test_3d)
    y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled).flatten()

    print("\n最终模型在测试集上的预测结果：")
    for i, (true_val, pred_val) in enumerate(zip(y_test_true, y_test_pred)):
        print(f"样本{i+15}：真实值={true_val:.4f}，预测值={pred_val:.4f}，误差={(pred_val - true_val):.4f}")

    # 4. 可视化：训练过程 & 预测结果
    # （可沿用之前“美化”代码，此处示例简单画一下）
    plt.figure(figsize=(10, 4), dpi=100)
    plt.subplot(1, 2, 1)
    plt.plot(history_opt.history['loss'], label='训练损失', linewidth=2, marker='o', markersize=4)
    plt.plot(history_opt.history['val_loss'], label='验证损失', linewidth=2, linestyle='--', marker='s', markersize=4)
    plt.title('最终模型损失曲线', fontsize=12)
    plt.xlabel('Epoch'); plt.ylabel('MSE'); plt.legend(); plt.grid(linestyle='--', alpha=0.5)

    plt.subplot(1, 2, 2)
    idxs = range(1, 18)
    # 训练集真实
    plt.plot(range(1, 15), y_train, 'bo-', label='训练集真实')
    # 测试集真实 & 预测
    for i in range(3):
        xi = 14 + i + 1
        # 实际目标
        plt.scatter(xi, y_test_true[i], color='g', marker='^', s=60, label='测试集真实' if i == 0 else "")
        # 预测值
        plt.scatter(xi, y_test_pred[i], color='r', marker='s', s=60, label='测试集预测' if i == 0 else "")
    # 连接线
    all_pred = np.concatenate([y_train, y_test_pred])
    plt.plot(range(1, 18), all_pred, 'r--', alpha=0.5, label='演示预测趋势')

    plt.title('最终模型预测结果', fontsize=12)
    plt.xlabel('样本编号'); plt.ylabel('里程 (10^4 km)'); plt.xticks(idxs)
    plt.legend(fontsize=8); plt.grid(linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig('../img/woa_cnn_final.png', dpi=100)
    plt.show()

    print("\n全部流程结束。")
