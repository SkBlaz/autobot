## helpers useful for storing the trained models etc.
import logging
import dill as pickle


def store_autobot_model(model_class, path, verbose=True):
    """
    A simple model storage method. It pickles a trained autoBOT object.
    
    :param model_class: The autoBOT object
    :param path: The path to the output file
    """

    f = open(path, "wb")
    pickle.dump(model_class, f)
    f.close()

    if verbose:
        logging.info("Stored the model info!")


def load_autobot_model(path, verbose=True):
    """
    Load a pickled autoBOT model.
    
    :param path: The path to the output file
    :param model_skeleton: The model object to be equipped with the stored model space
    :return unpickled_model: Returns a trained, unpickled model useful for further learning.
    """

    f = open(path, "rb")
    unpickled_model = pickle.load(f)
    f.close()

    return unpickled_model
