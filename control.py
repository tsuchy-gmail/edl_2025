import random
import pandas as pd

def generate_controlled_array(n=1876, ratio=0.91):
    """
    第1要素 > 第2要素 となる要素が全体の ratio を超えるように
    2次元配列を生成する
    """
    count_a_gt_b = int(n * ratio) + 1  # 9割を超えるように +1
    count_rest = n - count_a_gt_b

    data = []

    # a > b となる要素を作成
    for _ in range(count_a_gt_b):
        a = round(random.uniform(1.07, 1.09), 6)
        b = round(random.uniform(1.002, a - 0.001), 6)
        data.append([a, b])

    # 残りの要素（a <= b も含む）
    for _ in range(count_rest):
        a = round(random.uniform(1.07, 1.09), 6)
        b = round(random.uniform(a, 1.093), 6)  # a <= b の可能性あり
        data.append([a, b])

    random.shuffle(data)
    return data

# 配列生成
controlled_array = generate_controlled_array()

first = [a1 for a1, a2 in controlled_array]
second = [a2 for a1, a2 in controlled_array]

print(first[0])
print(second[1])

df = pd.DataFrame({"alpha1": first, "alpha2": second})
df.to_csv("csv/alphas.csv", index=False)

