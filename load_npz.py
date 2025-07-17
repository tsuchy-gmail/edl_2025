import numpy as np

CASE = "JMR2499"
embeds_save_path = f"saved_embeds/all_area/{CASE}_embeds.npy"
data = np.load(embeds_save_path)
print(data)
