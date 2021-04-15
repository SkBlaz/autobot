import pandas as pd
import matplotlib.pyplot as plt

dfx = pd.read_csv("historyFitness.tsv", sep="\t")
dfx2 = dfx.max(axis=0).values[1:]
plt.plot(dfx2)
plt.xlabel("Generation")
plt.ylabel("Fitness (F1)")
plt.savefig("fitness.png", dpi=300)
