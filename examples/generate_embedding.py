## What if we only wish to obtain the representation?
import autoBOTLib
import pandas as pd


def run():
    dataframe = pd.read_csv("../data/insults/train.tsv", sep="\t")
    train_sequences = dataframe['text_a']
    train_targets = dataframe['label']

    autoBOTLibObj = autoBOTLib.GAlearner(
        train_sequences, train_targets,
        time_constraint=0.1).evolve(representation_step_only=True)

    input_instance_embedding = autoBOTLibObj.transform(train_sequences)

    print(input_instance_embedding.shape)


if __name__ == "__main__":
    run()
