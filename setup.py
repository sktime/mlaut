from setuptools import setup


setup(name='mlaut',
      version='0.1.7',
      description='Automation of large-scale training, evaluation and benchmarking of machine learning algorithms.',
      url='https://alan-turing-institute.github.io/mlaut/',
      author='Viktor Kazakov',
      author_email='viktor.kazakov.18@ucl.ac.uk',
      long_description=open('README.txt').read(),
      license='BDS',
      packages=['mlaut', 
                'mlaut.analyze_results',
                'mlaut.data',
                'mlaut.estimators',
                'mlaut.experiments',
                'mlaut.shared'],
      install_requires=[
          'numpy',
          'h5py',
          'pandas',
          'scikit-learn',
          'scikit-posthocs', 
          'scipy', 
          'tables', 
          'tensorflow',
          'Orange3',
          'matplotlib',
          'wrapt'
      ],
      zip_safe=False)
