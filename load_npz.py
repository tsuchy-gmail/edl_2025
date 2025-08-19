import numpy as np

CASE = "JMR0020"
embeds_save_path = f"{CASE}_embeds.npz"
data = np.load(embeds_save_path)
print(data["embeds"].shape)
