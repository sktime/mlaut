Tutorial
========

The code below will help you get started with mleap.

The following Jupyter Notebook contains the code used below.

:download:`mleap.ipynb <../mleap.ipynb>`.


Data Preprocessing and Estimator Training Phase
-----------------------------------------------

**1. Get your data.**

The included code downloads preprocessed datasets from the UCI Machine Learning Repository and stores them locally.

:download:`delgado_datasets.py <../download_delgado/delgado_datasets.py>`

The enclosed code can be used as follows:  

.. literalinclude:: ../examples/basic.py
    :language: python
    :lines: 8-9

**2. Define Input and Output HDF5 objects.**

.. literalinclude:: ../examples/basic.py
    :language: python
    :lines: 13-14

**3. Store your data in HDF5 database.**

Once the datasets are stored locally we put them in an HDF5 database.

.. literalinclude:: ../examples/basic.py
    :language: python
    :lines: 17-20


**4. Split the data in test and train sets.**

.. literalinclude:: ../examples/basic.py
    :language: python
    :lines: 23-27


**5. Instantiate estimator objects and the experiments orchestrator class.**

.. literalinclude:: ../examples/basic.py
    :language: python
    :lines: 30-34

**6. Run the experiments.**

.. literalinclude:: ../examples/basic.py
    :language: python
    :lines: 37

At this point mleap will train all instantiated estimators on all datasets that were passed to the constructor. The trained models will be saved on the disk.

After this process is finished we can make predictions on the test sets using the trained estimators. in the previous step.

**7. Make predictions on the test sets.**

.. literalinclude:: ../examples/basic.py
    :language: python
    :lines: 40


Analyze Results of Machine Learning Experiments Phase
-----------------------------------------------------