from setuptools import setup


setup(name='mlaut',
      version='0.2.0',
      description='Automation of large-scale training, evaluation and benchmarking of machine learning algorithms.',
      url='https://alan-turing-institute.github.io/mlaut/',
      author='Viktor Kazakov',
      author_email='viktor.kazakov.18@ucl.ac.uk',
      long_description=open('README.txt').read(),
      license='BDS',
      packages=['mlaut',
                'mlaut.benchmarking',
                'mlaut.benchmarking.data',
                'mlaut.benchmarking.results',
                'mlaut.highlevel',
                'mlaut.resampling',
                'mlaut.shared',
                'mlaut.strategies'],
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
