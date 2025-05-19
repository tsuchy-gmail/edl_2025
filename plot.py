import pandas as pd
import matplotlib.pyplot as plt

# CSVファイルを読み込む（ファイル名は適宜変更）
df = pd.read_csv("result/test/2025_0508_154304_ok/test_loss_status")

# x軸用（エポックは全部100なのでインデックスを使う）
x = range(len(df))

# グラフのサイズとレイアウト設定
plt.figure(figsize=(15, 10))

# 1. loss 系
plt.subplot(3, 1, 1)
plt.plot(x, df['loss_inside'] * 2, label='loss_inside')
plt.plot(x, df['loss_outside'] * 2, label='loss_outside')
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

# 2. MSE 系
plt.subplot(3, 1, 2)
plt.plot(x, df['mse_inside'] * 2, label='mse_inside')
plt.plot(x, df['mse_outside'] * 2, label='mse_outside')
plt.title("Mean Squared Error")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.legend()
plt.grid(True)

# 3. KL 系
plt.subplot(3, 1, 3)
plt.plot(x, df['kl_inside'] * 2, label='kl_inside')
plt.plot(x, df['kl_outside'] * 2, label='kl_outside')
plt.title("KL Divergence")
plt.xlabel("Epoch")
plt.ylabel("KL")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("figure/test_test.png")

