import autoBOTLib
from autoBOTLib.optimization.optimization_utils import *
from autoBOTLib.optimization.optimization_feature_constructors import *
from autoBOTLib.optimization.optimization_engine import *
from autoBOTLib.misc.misc_helpers import *

import logging
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')

logging.getLogger().setLevel(logging.INFO)

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--classifier", default="autoBOT-base", type=str)
    parser.add_argument("--time", default=0.1, type=float)
    parser.add_argument("--popsize", default=8, type=int)
    parser.add_argument("--output_folder", default="results", type=str)
    parser.add_argument("--hof_size", default=3, type=int)
    parser.add_argument("--representation_type",
                        default="neurosymbolic",
                        type=str)
    parser.add_argument("--datafolder", default="./data/insults", type=str)
    parser.add_argument("--mutation_rate", default=0.3, type=float)
    parser.add_argument("--crossover_rate", default=0.6, type=float)
    parser.add_argument("--predict_data", default=None, type=str)
    parser.add_argument("--load_model", default=None, type=str)
    parser.add_argument("--num_cpu", default=8, type=int)
    parser.add_argument("--mode", default="learning", type=str)

    args = parser.parse_args()
    time_constraint = args.time

    import os

    directory = args.output_folder

    if not os.path.exists(directory):
        os.makedirs(directory)

    task = args.classifier

    logging.info("Starting to evaluate task: {}, learner {}".format(
        task, args.classifier))

    if args.mode == "prediction":

        cobj = load_autobot_model(f"{args.load_model}")
        test_data = pd.read_csv(args.predict_data, sep="\t")
        predictions = cobj.predict(test_data['text_a'])
        all_predictions = [str(x) for x in predictions]
        with open(args.output_folder + "/predictions_test.tsv", "w") as of:
            of.write("\n".join(all_predictions))

    elif args.mode == "learning":

        data_name = args.datafolder
        dataframe = pd.read_csv(f"{args.datafolder}", sep="\t")
        train_sequences = dataframe['text_a']
        train_targets = dataframe['label']

        autobotlibObj = autoBOTLib.GAlearner(
            train_sequences,
            train_targets,
            representation_type=args.representation_type,
            n_fold_cv=3,
            memory_storage="memory",
            sparsity=0.1,
            hof_size=args.hof_size,
            upsample=
            False,  ## Suitable for imbalanced data - randomized upsampling tends to help.
            time_constraint=args.time).evolve(
                nind=args.popsize,
                mutpb=args.mutation_rate,
                crossover_proba=args.crossover_rate,
                strategy="evolution")

        autobotlibObj.generate_report(output_folder=f"{args.output_folder}",
                                      job_id="as9y0gb98s")

        autoBOTLib.store_autobot_model(
            autoBOTObj, f"{args.output_folder}/autoBOTmodel.pickle")
