<p align="center">
  <img src="/docs/_images/logo.png" alt="mlaut" width="300px">
</p>
<p align="center">
  <a href="https://badge.fury.io/py/mlaut"><img src="https://badge.fury.io/py/mlaut.svg" alt="PyPI version" height="18"></a>
  <a href="https://opensource.org/licenses/BSD-3-Clause"><img src="https://img.shields.io/badge/License-BSD%203--Clause-blue.svg" alt="License"></a>
</p>


# mlaut (Machine Learning AUtomation Toolbox)

``mlaut`` is a modelling and workflow toolbox in python, written with the aim of simplifying large scale benchmarking of machine learning strategies, e.g., validation, evaluation and comparison with respect to predictive/task-specific performance or runtime. Key features are:

* Automation of the most common workflows for benchmarking modelling strategies on multiple datasets including statistical post-hoc analyses, with user-friendly default settings.

* Unified interface with support for scikit-learn strategies, keras deep neural network architectures, including easy user extensibility to (partially or completely) custom strategies.

* Higher-level meta-data interface for strategies, allowing easy specification of scikit-learn pipelines and keras deep network architectures, with user-friendly (sensible) default configurations.

* Easy setting up and loading of data set collections for local use (e.g., data frames from local memory, UCI repository, openML, Delgado study, PMLB).

* Back-end agnostic, automated local file system management of datasets, fitted models, predictions, and results, with the ability to easily resume crashed benchmark experiments with long running times.

List of [developers and contributors](AUTHORS.rst)

### Documentation

[<<<<<< Documentation available on alan-turing-institute.github.io/mlaut >>>>>>](https://alan-turing-institute.github.io/mlaut)

An example with the basic usage of ``mlaut`` can be found in the following [Jupyter Notebook](https://github.com/alan-turing-institute/mlaut/blob/master/examples/mlaut%20-%20Basic%20Usage.ipynb)

Please check the [examples directory](https://github.com/alan-turing-institute/mlaut/tree/master/examples) for more advanced use cases.

### Installing

Requires Python 3.6 or greater.

```
pip install mlaut
```

