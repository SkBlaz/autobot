autobot-cli --help

# "text_a" is the name of text field and "label" the name of labels
autobot-cli --train_data ./data/insults/train.tsv --output_folder CLI

# "text_a" is the name of the text field (labels are to be predicted and stored in the CLI folder)
autobot-cli --predict_data ./data/insults/test.tsv --output_folder CLI
