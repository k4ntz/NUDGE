import pandas as pd
import numpy as np


scores_human = pd.read_csv("human_getoutplus_log_0.csv")
h_mean = np.mean([arr.mean() for arr in np.array_split(scores_human, 4)])
h_std = np.std([arr.mean() for arr in np.array_split(scores_human, 4)])

h_max_m = np.mean([arr.max() for arr in np.array_split(scores_human, 4)])
h_max_std = np.std([arr.max() for arr in np.array_split(scores_human, 4)])

logic_mean = [pd.read_csv(f"logic_getoutplus_log_{i}.csv").mean() for i in [0, 2, 3, 4]]
logic_max = [pd.read_csv(f"logic_getoutplus_log_{i}.csv").max() for i in [0, 2, 3, 4]]

print("mean:")
print(f"NUDGE {np.mean(logic_mean):.2f} +- {np.std(logic_mean):.2f}")
print(f"HUMAN {h_mean:.2f} +- {h_std:.2f}")
print("max:")
print(f"NUDGE {np.mean(logic_max):.2f} +- {np.std(logic_max):.2f}")
print(f"HUMAN {h_max_m:.2f} +- {h_max_std:.2f}")