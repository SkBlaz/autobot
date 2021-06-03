Feature types
===============

.. list-table:: autoBOT's supported feature types
   :widths: 25 50 25
   :header-rows: 1

   * - Feature name
     - Description
     - Code (autoBOT)
   * - **Symbolic feature types**
     - 
     - 
   * - Concept features
     - Features derived from ConceptNet
     - concept_features
   * - Relational features - token
     - Features based on longer distances between tokens
     - relational_features_token
   * - Relational features - char
     - Features based on longer distances between characters
     - relational_features_char
   * - Topic features
     - KMeans clusters of TF-IDF into `d` dimensions, presences are feature values
     - topic_features
   * - Keywords
     - Keyword-based features
     - keyword_features
   * - Word-based features
     - Standard word TF-IDF-based features
     - word_features
   * - Character-based features
     - Standard character TF-IDF-based features
     - char_features
   * - Part-of-speech tag-based features
     - TF-IDF of documents, created based on POS annotations
     - pos_features
   * - **Sub-symbolic feature types**
     - 
     - 
   * - Neural - contextual
     - Features based on sentence-transformers library (XLM, multilingual)
     - contextual_features
   * - Neural - dbow
     - doc2vec - dbow
     - neural_features_dbow
   * - Neural - dm
     - doc2vec - dm
     - neural_features_dm
   * - Document graph
     - Features derived as node embeddings of the corpus graph
     - document_graph


Representation types are defined as follows (within autoBOT); the keys of these dictionaries are the arguments under the field `representation_type` (see `examples <https://github.com/SkBlaz/autobot/tree/master/examples>`_). Finally, note that new features are added on regular basis, hence there are more feature types available as in the original paper.

 .. code-block:: text

		 
    # Full stack
    feature_presets['neurosymbolic'] = [
	'concept_features', 'document_graph', 'neural_features_dbow',
	'neural_features_dm', 'relational_features_token', 'topic_features',
	'keyword_features', 'relational_features_char', 'char_features',
	'word_features', 'pos_features', 'contextual_features'
    ]

    # This one is ~language agnostic
    feature_presets['neurosymbolic-lite'] = [
	'document_graph', 'neural_features_dbow', 'neural_features_dm',
	'topic_features', 'keyword_features', 'relational_features_char', 'relational_features_token','char_features', 'word_features'
    ]

    # MLJ paper versions
    feature_presets['neurosymbolic-default'] = [
	'neural_features_dbow', 'neural_features_dm', 'keyword_features',
	'relational_features_char', 'char_features', 'word_features', "pos_features",
	'concept_features'
    ]

    feature_presets['neural'] = [
	'document_graph', 'neural_features_dbow', 'neural_features_dm'
    ]

    feature_presets['symbolic'] = [
	'concept_features', 'relational_features_token', 'topic_features',
	'keyword_features', 'relational_features_char', 'char_features',
	'word_features', 'pos_features'
    ]



So, for example, to use the `symbolic` feature space only, one can simply:

.. code:: python3

    autoBOTLibObj = autoBOTLib.GAlearner(train_sequences,
    train_targets,
    time_constraint = 1,
    representation_type = "symbolic").evolve()

    
