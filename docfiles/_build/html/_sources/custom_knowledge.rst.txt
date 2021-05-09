Using custom background knowledge
===============


One of the key novelties of *autoBOT* is the use of triplet-based graph databases as background knowledge (i.e. knowledge graphs). As shown in the examples, when doing the initialization:

.. code:: python3
	  
    autoBOTLibObj = autoBOTLib.GAlearner(
	    train_sequences,  # input sequences
		use_concept_features = True,
	    train_targets,  # target space 
	    memory_storage=
	    "./memory",  # tripled base for concept features
	    representation_type="neurosymbolic")  # or symbolic or neural

There is a dedicated parameter called *memory_storage*, which is a **path** that links to a file that contains the triplets like the following few examples:


 .. code-block:: text

    /a/[/r/CapableOf/,/c/en/context/,/c/en/matter_lot/]	/r/CapableOf	/c/en/context	/c/en/matter_lot	{"dataset": "/d/conceptnet/4/en", "license": "cc:by/4.0", "sources": [{"activity": "/s/activity/omcs/omcs1_possibly_free_text", "contributor": "/s/contributor/omcs/comdotatdotcom"}], "surfaceEnd": "matter a lot", "surfaceStart": "context", "surfaceText": "[[context]] can [[matter a lot]]", "weight": 1.0}
    /a/[/r/DerivedFrom/,/c/de/context/,/c/de/tex/n/]	/r/DerivedFrom	/c/de/context	/c/de/tex/n	{"dataset": "/d/wiktionary/de", "license": "cc:by-sa/4.0", "sources": [{"contributor": "/s/resource/wiktionary/de", "process": "/s/process/wikiparsec/2"}], "weight": 1.0}
    /a/[/r/DerivedFrom/,/c/en/acontextual/,/c/en/contextual/]	/r/DerivedFrom	/c/en/acontextual	/c/en/contextual	{"dataset": "/d/wiktionary/en", "license": "cc:by-sa/4.0", "sources": [{"contributor": "/s/resource/wiktionary/en", "process": "/s/process/wikiparsec/2"}], "weight": 1.0}


The database is formatted according to the `conceptnet <https://github.com/commonsense/conceptnet5/wiki/Downloads>`_. Note that you only need the first few columns of this file (subject-predicate-object). To use your own knowledge, simply provide a custom triplet database.
