import autoBOTLib
from autoBOTLib.optimization.optimization_utils import *
from autoBOTLib.optimization.optimization_feature_constructors import *
from autoBOTLib.optimization.optimization_engine import *
from autoBOTLib.misc.misc_helpers import *
import argparse
import os
import logging

# For the transformers not to cause problems
os.environ['TOKENIZERS_PARALLELISM'] = "false"

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')

logging.getLogger().setLevel(logging.INFO)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--time", default=0.1, type=float, help="Time in hours. Suggested value for real sota performances >= 1")
    parser.add_argument("--job_id", default="SomeRAnDoMJob9asd", type=str)
    parser.add_argument("--popsize", default=8, type=int)
    parser.add_argument("--output_folder", default="cli_results", type=str)
    parser.add_argument("--learner_preset", default="intense", type=str)
    parser.add_argument("--hof_size", default=1, type=int)
    parser.add_argument("--representation_type",
                        default="neurosymbolic",
                        type=str, help="Representation type. See the docs for more info.")
    parser.add_argument("--train_data", default="./data/insults/train.tsv", type=str)
    parser.add_argument("--mutation_rate", default=0.6, type=float)
    parser.add_argument("--crossover_rate", default=0.6, type=float)
    parser.add_argument("--predict_data", default=None, type=str)
    parser.add_argument("--load_model", default=None, type=str)
    parser.add_argument("--framework", default="scikit", type=str,
                        help="The computational ML back-end to use. Currently supports scikit (Default) and pyTorch (neural nets for sparse inputs)")
    parser.add_argument("--memory_storage", default="memory", type=str)
    parser.add_argument("--sparsity", default=0.05, type=float)
    parser.add_argument("--num_cpu", default=8, type=int, help="Number of threads to be used.")
    parser.add_argument("--n_folds", default=4, type=int)
    parser.add_argument("--upsample", default=False, type=bool, help="Minority class upsampling, might substantially boost performance!")
    parser.add_argument("--mode", default="learning", type=str, help="Learning or prediction. Prediction requires a path to a trained model.")

    args = parser.parse_args()
    time_constraint = args.time
    directory = args.output_folder
    
    logging.info("Starting the AutoML run, the used arguments are:")
    logging.info(args)
    
    if not os.path.exists(directory):
        os.makedirs(directory)

    if args.mode == "prediction":

        cobj = load_autobot_model(f"{args.load_model}")
        test_data = pd.read_csv(args.predict_data, sep="\t")
        predictions = cobj.predict(test_data['text_a'])
        all_predictions = [str(x) for x in predictions]
        pred_path = os.path.join(args.output_folder, "predictions.txt")
        with open(pred_path, "w") as of:
            of.write("\n".join(all_predictions))
        logging.info(f"Wrote predictions to {pred_path}")

    elif args.mode == "learning":

        dataframe = pd.read_csv(f"{args.train_data}", sep="\t").iloc[:]
        train_sequences = dataframe['text_a'].values.tolist()
        train_targets = dataframe['label'].values
        autobotlibObj = autoBOTLib.GAlearner(
            train_sequences,
            train_targets,
            representation_type=args.representation_type,
            n_fold_cv=args.n_folds,
            memory_storage=args.memory_storage,
            sparsity=args.sparsity,
            learner_preset=args.learner_preset,
            hof_size=args.hof_size,
            framework=args.framework,
            upsample=args.upsample,
            time_constraint=args.time).evolve(
                nind=args.popsize,
                mutpb=args.mutation_rate,
                crossover_proba=args.crossover_rate,
                strategy="evolution")

        logging.info(f"Storing the model as {args.output_folder}/autoBOTmodel.pickle")
        
        autoBOTLib.store_autobot_model(
            autobotlibObj, f"{args.output_folder}/autoBOTmodel.pickle")

        try:
            autobotlibObj.generate_report(output_folder=f"{args.output_folder}",
                                          job_id=args.job_id)
            
        except Exception as es:
            logging.info(f"Could not produce the report. {es}")


if __name__ == "__main__":
    main()
