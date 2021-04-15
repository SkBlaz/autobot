## minimal test
from deap import base, creator
creator.create("FitnessMulti", base.Fitness, weights=(1.0, ))
creator.create("Individual", list, fitness=creator.FitnessMulti)
import dill as pickle


def store_model(model_class, path, verbose=True):

    f = open(path, "wb")
    pickle.dump(model_class, f)
    f.close()

    if verbose:
        print("Stored the model info!")


def load_model(path, verbose=True):

    f = open(path, "rb")
    unpickled_model = pickle.load(f)
    f.close()

    return unpickled_model


if __name__ == "__main__":

    model = load_model("test.pickle")
    print(model)
