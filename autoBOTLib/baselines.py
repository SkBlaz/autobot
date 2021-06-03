## baseline learners

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn import pipeline
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from gensim.utils import simple_preprocess
try:
    import nltk
    nltk.data.path.append("./nltk_data")
except Exception as es:
    import nltk
    print(es)
import pandas as pd
from sklearn.pipeline import FeatureUnion
from sklearn import pipeline
from sklearn.preprocessing import Normalizer
import numpy as np
from sklearn.metrics import f1_score
import logging
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')

try:
    from tpot import TPOTClassifier
except:
    print("no tpot")

try:
    from simpletransformers.classification import ClassificationModel
    import pandas as pd
    import logging

except:
    print("No simpletransformers lib")

global_svm_range = [0.1, 0.5, 1, 5, 10, 20, 50, 100, 500]


def get_svm_char_pipeline(train_sequences,
                          dev_sequences,
                          train_targets,
                          dev_targets,
                          time_constraint=1,
                          num_cpu=1,
                          max_features=1000):
    copt = 0
    opt_c = 0
    for c in global_svm_range:
        logging.info("Testing c value of {}".format(c))
        vectorizer = TfidfVectorizer(analyzer='char',
                                     ngram_range=(2, 4),
                                     max_features=max_features)
        clf = LinearSVC(C=c)
        svm_pip = pipeline.Pipeline([('vec', vectorizer),
                                     ('scale', Normalizer()),
                                     ('classifier', clf)])
        svm_pip.fit(train_sequences, train_targets)
        preds = svm_pip.predict(dev_sequences)
        if len(np.unique(train_targets)) > 2:
            average = "micro"
        else:
            average = "binary"
        f1 = f1_score(preds, dev_targets, average=average)
        if f1 > copt:
            logging.info("Improved performance to {}".format(f1))
            copt = f1
            opt_c = c
    vectorizer = TfidfVectorizer(analyzer='char',
                                 ngram_range=(2, 4),
                                 max_features=max_features)
    clf = LinearSVC(C=opt_c)
    svm_pip = pipeline.Pipeline([('vec', vectorizer), ('scale', Normalizer()),
                                 ('classifier', clf)])
    return svm_pip.fit(train_sequences, train_targets)


def get_svm_word_pipeline(train_sequences,
                          dev_sequences,
                          train_targets,
                          dev_targets,
                          time_constraint=1,
                          num_cpu=1,
                          max_features=1000):
    copt = 0
    opt_c = 0
    for c in global_svm_range:
        logging.info("Testing c value of {}".format(c))
        vectorizer = TfidfVectorizer(ngram_range=(1, 3),
                                     max_features=max_features)
        clf = LinearSVC(C=c)
        svm_pip = pipeline.Pipeline([('vec', vectorizer),
                                     ('scale', Normalizer()),
                                     ('classifier', clf)])
        svm_pip.fit(train_sequences, train_targets)
        preds = svm_pip.predict(dev_sequences)
        if len(np.unique(train_targets)) > 2:
            average = "micro"
        else:
            average = "binary"
        f1 = f1_score(preds, dev_targets, average=average)
        if f1 > copt:
            logging.info("Improved performance to {}".format(f1))
            copt = f1
            opt_c = c
    vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=max_features)
    clf = LinearSVC(C=opt_c)
    svm_pip = pipeline.Pipeline([('vec', vectorizer), ('scale', Normalizer()),
                                 ('classifier', clf)])
    return svm_pip.fit(train_sequences, train_targets)


def get_lr_word_pipeline(train_sequences,
                         dev_sequences,
                         train_targets,
                         dev_targets,
                         time_constraint=1,
                         num_cpu=1,
                         max_features=1000):
    copt = 0
    opt_c = 0
    for c in global_svm_range:
        logging.info("Testing c value of {}".format(c))
        vectorizer = TfidfVectorizer(ngram_range=(1, 3),
                                     max_features=max_features)
        clf = LogisticRegression(C=c)
        lr_pip = pipeline.Pipeline([('vec', vectorizer),
                                    ('scale', Normalizer()),
                                    ('classifier', clf)])
        lr_pip.fit(train_sequences, train_targets)
        preds = lr_pip.predict(dev_sequences)
        if len(np.unique(train_targets)) > 2:
            average = "micro"
        else:
            average = "binary"
        f1 = f1_score(preds, dev_targets, average=average)
        if f1 > copt:
            logging.info("Improved performance to {}".format(f1))
            copt = f1
            opt_c = c
    vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=max_features)
    clf = LinearSVC(C=opt_c)
    lr_pip = pipeline.Pipeline([('vec', vectorizer), ('scale', Normalizer()),
                                ('classifier', clf)])
    return lr_pip.fit(train_sequences, train_targets)


def get_lr_char_pipeline(train_sequences,
                         dev_sequences,
                         train_targets,
                         dev_targets,
                         time_constraint=1,
                         num_cpu=1,
                         max_features=1000):
    copt = 0
    opt_c = 0
    for c in global_svm_range:
        logging.info("Testing c value of {}".format(c))
        vectorizer = TfidfVectorizer(analyzer='char',
                                     ngram_range=(2, 4),
                                     max_features=max_features)
        clf = LogisticRegression(C=c)
        lr_pip = pipeline.Pipeline([('vec', vectorizer),
                                    ('scale', Normalizer()),
                                    ('classifier', clf)])
        lr_pip.fit(train_sequences, train_targets)
        preds = lr_pip.predict(dev_sequences)
        if len(np.unique(train_targets)) > 2:
            average = "micro"
        else:
            average = "binary"
        f1 = f1_score(preds, dev_targets, average=average)

        if f1 > copt:
            logging.info("Improved performance to {}".format(f1))
            copt = f1
            opt_c = c
    vectorizer = TfidfVectorizer(analyzer='char',
                                 ngram_range=(2, 4),
                                 max_features=max_features)
    clf = LogisticRegression(C=opt_c)
    lr_pip = pipeline.Pipeline([('vec', vectorizer), ('scale', Normalizer()),
                                ('classifier', clf)])
    return lr_pip.fit(train_sequences, train_targets)


def get_lr_word_char_pipeline(train_sequences,
                              dev_sequences,
                              train_targets,
                              dev_targets,
                              time_constraint=1,
                              num_cpu=1,
                              max_features=1000):
    copt = 0
    opt_c = 0
    for c in global_svm_range:
        logging.info("Testing c value of {}".format(c))
        vectorizer = TfidfVectorizer(analyzer='char',
                                     ngram_range=(2, 4),
                                     max_features=max_features)
        vectorizer2 = TfidfVectorizer(ngram_range=(1, 3),
                                      max_features=max_features)

        features = [('word', vectorizer), ('char', vectorizer2)]
        clf = LogisticRegression(C=c)
        lr_pip = pipeline.Pipeline([('union',
                                     FeatureUnion(transformer_list=features)),
                                    ('scale', Normalizer()),
                                    ('classifier', clf)])

        lr_pip.fit(train_sequences, train_targets)
        preds = lr_pip.predict(dev_sequences)
        if len(np.unique(train_targets)) > 2:
            average = "micro"
        else:
            average = "binary"
        f1 = f1_score(preds, dev_targets, average=average)
        if f1 > copt:
            logging.info("Improved performance to {}".format(f1))
            copt = f1
            opt_c = c

    vectorizer = TfidfVectorizer(analyzer='char',
                                 ngram_range=(2, 4),
                                 max_features=max_features)
    vectorizer2 = TfidfVectorizer(ngram_range=(1, 3),
                                  max_features=max_features)
    features = [('word', vectorizer), ('char', vectorizer2)]
    clf = LogisticRegression(C=opt_c)
    lr_pip = pipeline.Pipeline([('union',
                                 FeatureUnion(transformer_list=features)),
                                ('scale', Normalizer()), ('classifier', clf)])
    return lr_pip.fit(train_sequences, train_targets)


def get_svm_word_char_pipeline(train_sequences,
                               dev_sequences,
                               train_targets,
                               dev_targets,
                               time_constraint=1,
                               num_cpu=1,
                               max_features=1000):
    copt = 0
    for c in global_svm_range:
        logging.info("Testing c value of {}".format(c))
        vectorizer = TfidfVectorizer(analyzer='char',
                                     ngram_range=(2, 4),
                                     max_features=max_features)
        vectorizer2 = TfidfVectorizer(ngram_range=(1, 3),
                                      max_features=max_features)

        features = [('word', vectorizer), ('char', vectorizer2)]
        clf = LinearSVC(C=c)
        svm_pip = pipeline.Pipeline([('union',
                                      FeatureUnion(transformer_list=features)),
                                     ('scale', Normalizer()),
                                     ('classifier', clf)])

        svm_pip.fit(train_sequences, train_targets)
        preds = svm_pip.predict(dev_sequences)
        if len(np.unique(train_targets)) > 2:
            average = "micro"
        else:
            average = "binary"
        f1 = f1_score(preds, dev_targets, average=average)
        if f1 > copt:
            logging.info("Improved performance to {}".format(f1))
            copt = f1

    vectorizer = TfidfVectorizer(analyzer='char',
                                 ngram_range=(2, 4),
                                 max_features=max_features)
    vectorizer2 = TfidfVectorizer(ngram_range=(1, 3),
                                  max_features=max_features)
    features = [('word', vectorizer), ('char', vectorizer2)]
    clf = LinearSVC(C=c)
    svm_pip = pipeline.Pipeline([('union',
                                  FeatureUnion(transformer_list=features)),
                                 ('scale', Normalizer()), ('classifier', clf)])
    return svm_pip.fit(train_sequences, train_targets)


def get_tpot_word_pipeline(train_sequences,
                           dev_sequences,
                           train_targets,
                           dev_targets,
                           time_constraint=1,
                           num_cpu=1,
                           max_features=1000):

    vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=max_features)
    features = [('word', vectorizer)]
    clf = TPOTClassifier(generations=5,
                         population_size=50,
                         verbosity=2,
                         random_state=42)

    auml_pip = pipeline.Pipeline([('union',
                                   FeatureUnion(transformer_list=features)),
                                  ('scale', Normalizer())])

    sequence_space = train_sequences.tolist() + dev_sequences.tolist()

    X_train = auml_pip.fit_transform(sequence_space)
    Y_train = np.array(train_targets.tolist() + dev_targets.tolist())

    clf.fit(X_train.todense(), Y_train)
    return (auml_pip, clf)


def get_majority(train_sequences,
                 dev_sequences,
                 train_targets,
                 dev_targets,
                 time_constraint=1,
                 num_cpu=1,
                 max_features=1000):
    copt = 0
    c = 0
    max_features = 100
    logging.info("Testing c value of {}".format(c))
    vectorizer = TfidfVectorizer(analyzer='char',
                                 ngram_range=(1, 1),
                                 max_features=max_features)
    vectorizer2 = TfidfVectorizer(ngram_range=(1, 1),
                                  max_features=max_features)

    features = [('word', vectorizer), ('char', vectorizer2)]
    clf = DummyClassifier()
    svm_pip = pipeline.Pipeline([('union',
                                  FeatureUnion(transformer_list=features)),
                                 ('scale', Normalizer()), ('classifier', clf)])

    svm_pip.fit(train_sequences, train_targets)
    preds = svm_pip.predict(dev_sequences)
    if len(np.unique(train_targets)) > 2:
        average = "micro"
    else:
        average = "binary"
    f1 = f1_score(preds, dev_targets, average=average)
    if f1 > copt:
        logging.info("Improved performance to {}".format(f1))
        copt = f1

    vectorizer = TfidfVectorizer(analyzer='char',
                                 ngram_range=(1, 1),
                                 max_features=max_features)
    vectorizer2 = TfidfVectorizer(ngram_range=(1, 1),
                                  max_features=max_features)
    features = [('word', vectorizer), ('char', vectorizer2)]
    clf = DummyClassifier()
    svm_pip = pipeline.Pipeline([('union',
                                  FeatureUnion(transformer_list=features)),
                                 ('scale', Normalizer()), ('classifier', clf)])
    return svm_pip.fit(train_sequences, train_targets)


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)


def get_doc2vec(train_sequences,
                dev_sequences,
                train_targets,
                dev_targets,
                time_constraint=1,
                num_cpu=1,
                max_features=1000,
                classifier="LR"):

    total_sequences_training = train_sequences.values.tolist(
    ) + dev_sequences.values.tolist()
    total_labels_training = train_targets.tolist() + dev_targets.tolist()
    if classifier == "LR":
        clf = LogisticRegression()
    else:
        clf = LinearSVC()
    documents = [
        TaggedDocument(simple_preprocess(doc), [i])
        for i, doc in enumerate(total_sequences_training)
    ]
    model = Doc2Vec(documents,
                    vector_size=512,
                    window=5,
                    min_count=2,
                    epochs=32,
                    num_cpu=8)
    vecs = []
    for doc in total_sequences_training:
        vector = model.infer_vector(simple_preprocess(doc))
        vecs.append(vector)
    train_matrix = np.matrix(vecs)
    clf.fit(train_matrix, total_labels_training)
    return (clf, model)


def get_bert_base(train_sequences,
                  dev_sequences,
                  train_targets,
                  dev_targets,
                  time_constraint=1,
                  num_cpu=1,
                  max_features=1000,
                  model="bert-base"):

    'text' 'labels'
    total_sequences_training = train_sequences.values.tolist(
    ) + dev_sequences.values.tolist()

    total_labels_training = train_targets.tolist() + dev_targets.tolist()

    train_df = pd.DataFrame()
    train_df['text'] = total_sequences_training
    train_df['labels'] = total_labels_training

    # Create a ClassificationModel
    if model == "bert-base":
        model = ClassificationModel('bert',
                                    'bert-base-cased',
                                    num_labels=len(set(total_labels_training)),
                                    args={
                                        'reprocess_input_data': True,
                                        'overwrite_output_dir': True
                                    },
                                    use_cuda=True)

    elif model == "roberta-base":
        model = ClassificationModel('roberta',
                                    'roberta-base',
                                    num_labels=len(set(total_labels_training)),
                                    args={
                                        'reprocess_input_data': True,
                                        'overwrite_output_dir': True
                                    },
                                    use_cuda=True)

    model.args['num_train_epochs'] = 20
    model.args['max_sequence_length'] = 512
    model.args['save_eval_checkpoints'] = False
    model.args['save_model_every_epoch'] = False
    model.args['save_steps'] = 4000
    model.train_model(train_df)
    return model
