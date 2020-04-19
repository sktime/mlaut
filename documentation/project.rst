Description of the project
==========================


The goal of ``mlaut`` is to automate the workflow for training and comparing machine learning estimators. The package facilitates the training of a large number of estimators on multiple datasets. It also provides a statistical framework for comparing the performance of the trained estimators.


Overview
--------

``mlaut`` seeks to expand the functionality of the `scikit-learn <http://scikit-learn.org>`_ package. It also aims at providing a seamless integration with other packages such as `scipy <https://www.scipy.org/>`_, `statsmodels <https://www.statsmodels.org>`_, `scikit-posthocs <https://github.com/maximtrp/scikit-posthocs>`_, `Tensorflow <https://www.tensorflow.org/>`_ and `Orange <https://orange.biolab.si/>`_.

.. note:: Knowledge of all above mentioned packages is not necessarily required in order to work with ``mlaut``. However, in order to make full use of it and be able to expand its functionality understanding of `scikit-learn <http://scikit-learn.org>`_ is highly desirable. 

``mlaut`` is comprised of the following main modules.

Applications and use
---------------------

``mlaut`` can be used for running supervised classification and regression experiments. The package currently provides an interface to scikit-learn and keras models  but can easily be extended by the user to incorporate additional toolboxes.
	
``mlaut`` is suitable for creating a begin-to-end pipeline for processing data, training machine learning experiments, making predictions and applying statistical tests to benchmark the performance of the different models. 
	
Specifically, ``mlaut`` can be used to:

* Automate the entire workflow for large-scale machine learning experiments studies. This includes structuring and transforming the data, selecting the appropriate estimators for the task and data data at hand, tuning the estimators and finally comparing the results.

* Fit data and make predictions by using the prediction strategies as described in \ref{subsection:estimators} or by implementing new prediction strategies.

* Evaluate the results of the prediction strategies in a uniform and statistically sound manner.

