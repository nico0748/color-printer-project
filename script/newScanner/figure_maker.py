import matplotlib.pyplot as plt

# ====== 入力データ（ここを書き換える） ======
mse_values = [
    842.15, 803.42,765.71, 694.55, 613.67
]
# ============================================

# 試行回数（1,2,3,...）
trials = list(range(1, len(mse_values) + 1))

# 最大・最小値とそのインデックス
max_mse = max(mse_values)
min_mse = min(mse_values)
max_idx = mse_values.index(max_mse)
min_idx = mse_values.index(min_mse)

# プロット
plt.figure()
plt.plot(trials, mse_values, marker='o', linestyle='-')

# 最大値・最小値を強調表示
plt.scatter(trials[max_idx], max_mse, s=100)
plt.scatter(trials[min_idx], min_mse, s=100)

# 注釈（テキスト）
plt.annotate(
    f"Max: {max_mse}",
    (trials[max_idx], max_mse),
    textcoords="offset points",
    xytext=(10, 10)
)

plt.annotate(
    f"Min: {min_mse}",
    (trials[min_idx], min_mse),
    textcoords="offset points",
    xytext=(10, -15)
)

# ラベル・タイトル
plt.xlabel("Trial Number")
plt.ylabel("MSE")
plt.title("NN = 80 Hidden Units - MSE over Trials")

plt.grid(True)
plt.tight_layout()
plt.show()