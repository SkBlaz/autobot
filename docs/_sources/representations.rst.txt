Obtaining underlying representations
===============
Obtaining the representations of documents so you can explore potentially different learning schemes is discussed in the following example:

.. code:: python3

    import autoBOTLib
    import pandas as pd

    dataframe = pd.read_csv("../data/insults/train.tsv", sep="\t")
    train_sequences = dataframe['text_a']
    train_targets = dataframe['label']

    autoBOTLibObj = autoBOTLib.GAlearner(
        train_sequences,
        train_targets,
        time_constraint=0.1).evolve()

    input_instance_embedding = autoBOTLibObj.transform(train_sequences)

    print(input_instance_embedding.shape)


Note that as long as the *evolve()* was called, the *transform()* method is able to use the trained vectorizers to obtain sparse (Scipy) matrices.
