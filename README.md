![skpro](/docs/_images/logo.png)
<p align="center">
  <a href="https://badge.fury.io/py/mlaut"><img src="https://badge.fury.io/py/mlaut.svg" alt="PyPI version" height="18"></a>
  <a href="https://opensource.org/licenses/BSD-3-Clause"><img src="https://img.shields.io/badge/License-BSD%203--Clause-blue.svg" alt="License"></a>
</p>


# mlaut (Machine Learning AUtomation Toolbox)

MLAUT is a modelling and workflow toolbox in python, written with the aim of simplifying large scale benchmarking of machine learning strategies, e.g., validation, evaluation and comparison with respect to predictive/task-specific performance or runtime. Key features are:

* Automation of the most common workflows for benchmarking modelling strategies on multiple datasets including statistical post-hoc analyses, with user-friendly default settings.

* Unified interface with support for scikit-learn strategies, keras deep neural network architectures, including easy user extensibility to (partially or completely) custom strategies.

* Higher-level meta-data interface for strategies, allowing easy specification of scikit-learn pipelines and keras deep network architectures, with user-friendly (sensible) default configurations.

* Easy setting up and loading of data set collections for local use (e.g., data frames from local memory, UCI repository, openML, Delgado study, PMLB).

* Back-end agnostic, automated local file system management of datasets, fitted models, predictions, and results, with the ability to easily resume crashed benchmark experiments with long running times.

### Documentation

[<<<<<< Documentation available on kiraly-group.github.io/mlaut/ >>>>>>](https://kiraly-group.github.io/mlaut/)

### Installing

Requires Python 3.6 or greater.

```
pip install mlaut
```


