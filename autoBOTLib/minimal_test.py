## minimal test
import random
from deap import base, creator, tools
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


class TestClass:
    def __init__(self):
        self.toolbox = base.Toolbox()
        self.total_params = 5
        self.base = base

    def mutReg(self, individual, p=1):
        """
        Custom mutation operator used for regularization optimization.

        :param individual: individual (vector of floats)
        :return individual: An individual solution.
        """

        individual[0] += random.random() * self.reg_constant
        return individual,

    def somefun(self):

        self.toolbox.register("attr_float", random.uniform, 0.00001, 0.999999)

        self.toolbox.register("attr_float", random.uniform, 0.00001, 0.999999)
        self.toolbox.register("individual",
                              tools.initRepeat,
                              creator.Individual,
                              self.toolbox.attr_float,
                              n=self.total_params)
        self.toolbox.register("population",
                              tools.initRepeat,
                              list,
                              self.toolbox.individual,
                              n=100)
        self.toolbox.register("mate", tools.cxUniform, indpb=0.5)
        self.toolbox.register("mutate",
                              tools.mutGaussian,
                              mu=0,
                              sigma=0.2,
                              indpb=0.2)
        self.toolbox.register("mutReg", self.mutReg)
        self.toolbox.register("select", tools.selTournament)


clx = TestClass()
clx.somefun()

store_model(clx, "test.pickle")

model = load_model("test.pickle")

print(model)
