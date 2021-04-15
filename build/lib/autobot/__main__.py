from autobot.data_utils import *
from sklearn import preprocessing
from autobot.feature_constructors import *
from autobot.strategy_ga import *
from autobot.helpers import *

import logging
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logging.getLogger().setLevel(logging.INFO)


def evaluate_model(train,
                   dev,
                   time_constraint=0.1,
                   learner="autoBOT-ga-big",
                   max_features=1000,
                   task_name="run:",
                   popsize=30,
                   hof_size=1,
                   crossover_rate=0.6,
                   mutation_rate=0.1):

    train_sequences = train['text_a']
    dev_sequences = dev['text_a']
    print("evaluating {}".format(learner))

    train_targets = train['label']
    dev_targets = dev['label']

    if learner == "autoBOT-base":
        train_sequences = pd.concat([train_sequences, dev_sequences], axis=0)
        train_targets = np.concatenate((train_targets, dev_targets))
        pipeline_obj = GAlearner(train_sequences,
                                 train_targets,
                                 time_constraint=time_constraint,
                                 num_cpu=args.num_cpu,
                                 task_name=task_name,
                                 hof_size=hof_size,
                                 representation_type=args.representation_type)

        pipeline_obj.evolve(nind=popsize,
                            strategy="evolution",
                            crossover_proba=crossover_rate,
                            mutpb=mutation_rate)

    return pipeline_obj


def write_to_file(fname, string):
    with open(fname, "w") as of:
        of.write(string)


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
        data_processor = genericProcessor()  ## load a generic file processor
        data_name = args.datafolder

        train_examples = data_processor.get_train_examples(data_name)
        dev_examples = data_processor.get_dev_examples(data_name)
        test_examples = data_processor.get_test_examples(data_name)

        cobj = evaluate_model(train_examples,
                              dev_examples,
                              time_constraint,
                              learner=args.classifier,
                              task_name=task,
                              popsize=args.popsize,
                              hof_size=args.hof_size,
                              mutation_rate=args.mutation_rate,
                              crossover_rate=args.crossover_rate)

        store_autobot_model(cobj, f"{args.output_folder}/autoBOTmodel.pickle")

        feature_importances = None
        transformer_tag = False

        test_df = test_examples['text_a'].values.tolist()
        predictions = cobj.predict(test_df)
        real = test_examples['label']

        try:

            ## performance report first
            performance = compute_metrics(task, predictions, real)
            perf = [
                performance['acc'], performance['f1'],
                performance['acc_and_f1']
            ]
            perf = [str(x) for x in perf]
            names = "\t".join(['Accuracy', 'F1', 'Acc_and_F1']) + "\n"
            ostr = "\t".join(perf)
            final = names + ostr
            write_to_file(args.output_folder + "/performance.txt", final)

            ## store fitness strings
            fitness_string = cobj.fitness_container
            fdf = []
            ids = []
            for enx, el in enumerate(fitness_string):
                ids.append(enx)
                parts = [str(x[0]) for x in el]
                fdf.append(parts)

            dfx = pd.DataFrame(fdf)
            dfx = dfx.T
            dfx.columns = ids
            dfx.to_csv(args.output_folder + "/fitness_space.txt", sep="\t")

            ## store global importances
            feature_importances = cobj.hof[
                0]  ## global importances .feature_names rabi
            struct = []

            for a, b in zip(feature_importances, cobj.feature_names):
                struct.append((str(a), str(b)))

            dfx = pd.DataFrame(struct)
            dfx.columns = ['Importance', 'Feature subspace']
            dfx.to_csv(args.output_folder + "/subspace_ranking.txt", sep="\t")

            ## store global top features
            feature_ranking = cobj.global_feature_map  ## all features
            feature_ranking.to_csv(args.output_folder + "/top_features.txt",
                                   sep="\t")

            final_learners = cobj.summarise_final_learners()
            pd.to_csv(args.output_folder + "/learners.tsv", sep="\t")

        except Exception as es:
            print(es)
