Description of the project
==========================


The goal of ``mleap`` is to automate the workflow for training and comparing 
machine learning estimators. The package facilitates the training of a large 
number of estimators on multiple datasets. 
It also provides a statistical framework for comparing the performance of the 
trained estimators.


Overview
--------

``mleap`` seeks to expand the functionality of the 
`scikit-learn <http://scikit-learn.org>`_ package. It also aims at providing a
seamless integration with other packages such as `scipy <https://www.scipy.org/>`_,  
`statsmodels <https://www.statsmodels.org>`_ and `scikit-posthocs <https://github.com/maximtrp/scikit-posthocs>`_.

.. note:: Knowledge of all above mentioned packages is not necessarily required 
          in order to work with ``mleap``. However, in order to make full use of it and be
          able to expand its functionality understanding of `scikit-learn <http://scikit-learn.org>`_
          is highly desirable.

``mleap`` is comprised of the following main modules.

* :ref:`estimators`
* :ref:`data`
* :ref:`experiments`
* :ref:`analyze_results`
* :ref:`shared`

We will explain on a high level the interaction between the modules.