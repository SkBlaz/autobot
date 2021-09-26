## Train an autoBOT classifier
mkdir results;
python autoBOTLib --mode learning --datafolder ./data/insults/train.tsv --output_folder results --classifier autoBOT-base --time 0.1 --hof_size 3 --representation_type neurosymbolic-lite --mutation_rate 0.3 --crossover_rate 0.6

## Obtain predictions from a trained model
python autoBOTLib --mode prediction --load_model ./results/autoBOTmodel.pickle --predict_data ./data/insults/test.tsv
