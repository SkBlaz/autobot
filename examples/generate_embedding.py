## What if we only wish to obtain the representation?
import autoBOTLib
import pandas as pd


def run():
    dataframe = pd.read_csv("../data/insults/train.tsv", sep="\t")
    train_sequences = dataframe['text_a'].iloc[0:20]
    train_targets = dataframe['label'].iloc[0:20]

    autoBOTLibObj = autoBOTLib.GAlearner(
        train_sequences, train_targets,
        time_constraint=0.1).evolve(representation_step_only=True)

    input_instance_embedding = autoBOTLibObj.transform(train_sequences)

    all_feature_names = []
    for transformer in autoBOTLibObj.vectorizer.named_steps[
            'union'].transformer_list:
        features = transformer[1].steps[1][1].get_feature_names()
        all_feature_names += features

    assert input_instance_embedding.shape[1] == len(all_feature_names)

    print(input_instance_embedding.shape)


if __name__ == "__main__":
    run()
