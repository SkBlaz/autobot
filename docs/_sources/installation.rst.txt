Installation
===============

To install the stable version of the code, you can try:

.. code:: bash

    pip install autoBOTLib

Or for the current master branch

.. code:: bash

    pip install git+https://github.com/SkBlaz/autobot

Or directly from the repo

.. code:: bash

   python setup.py install

Requirements can be installed via (from the repo folder):

.. code:: bash

    pip install -r requirements.txt

We also provide the conda virtual env specification file (yml), found in the root directory of `autoBOT <https://github.com/SkBlaz/autobot/>`_ repo.

.. code:: bash
	  
   conda env create -f conda_env.yml
   
To test whether the core library functionality works well, you can run the test suite from the ./tests folder (`autoBOT <https://github.com/SkBlaz/autobot/>`_) as::
  
  py.test
