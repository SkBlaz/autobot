import os
import logging
import nltk

# Configure logging first
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logging.getLogger().setLevel(logging.INFO)

# Set environment variables
os.environ['TOKENIZERS_PARALLELISM'] = "false"

# Download NLTK resources
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)

# Import all module functionality
from autoBOTLib.features.features_keyword import *
from autoBOTLib.features.features_contextual import *
from autoBOTLib.features.features_token_relations import *
from autoBOTLib.features.features_concepts import *
from autoBOTLib.features.features_document_graph import *
from autoBOTLib.features.features_sentence_embeddings import *
from autoBOTLib.features.features_topic import *

from autoBOTLib.optimization.optimization_utils import *
from autoBOTLib.optimization.optimization_feature_constructors import *
from autoBOTLib.optimization.optimization_engine import *
from autoBOTLib.misc.misc_helpers import *
