## What if we only wish to obtain the representation?
import autoBOTLib
import pandas as pd
import umap
import matplotlib.pyplot as plt
import seaborn as sns

def run():
    dataframe = pd.read_csv("../data/depression/train.tsv", sep="\t")
    train_sequences = dataframe['text_a'][0:]
    train_targets = dataframe['label'][0:]

    autoBOTLibObj = autoBOTLib.GAlearner(
        train_sequences, train_targets,
        time_constraint=0.1).evolve(representation_step_only=True)

    input_instance_embedding = autoBOTLibObj.transform(train_sequences)

    print(input_instance_embedding.shape)
    transf = umap.UMAP()
    embedding = transf.fit_transform(input_instance_embedding)
    sns.scatterplot(
        embedding[:, 0],
        embedding[:, 1],
        hue=train_targets, palette="coolwarm")
    plt.gca().set_aspect('equal', 'datalim')
    plt.title(f'UMAP-based document projection ({input_instance_embedding.shape[1]}D -> 2D)', fontsize=12)
    plt.show() #or store with plt.savefig("path.pdf", dpi=300)

if __name__ == "__main__":
    run()
