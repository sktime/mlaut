# Machine Learning Experiemnts Study

Package that automates running machine learning experiments.

### Installing

Requires Python 3.6.


Create virtual environment:
```
virtualenv -p python3 py36
```
Activate virtual environment and install dependencies:
```
source ./py36/bin/activate
pip install -r requirements.txt
```

### Run Code

```
python main.py
```

### Change number of CPU cores

Edit variable: 
```
GRIDSEARCH_CV_NUM_PARALLEL_JOBS in ./src/static_variables.py 
```
to change the number of CPU cores available for the calculations.

 

