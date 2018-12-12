from setuptools import setup


setup(name='mlaut',
      version='0.1.2',
      description='Automation of large-scale training, evaluation and benchmarking of machine learning algorithms.',
      url='https://viktorkaz.github.io/',
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
          'numpy>=1.7',
          'h5py>=2.8'
          'pandas>=0.21',
          'scikit-learn>=0.20.1',
          'scikit-posthocs>=0.3.4', 
          'scipy>=1.0.0', 
          'tables>=3.4.2', 
          'tensorflow>=1.5.0',
          'Orange3>=3.18.0',
          'matplotlib>=2.1.0'
      ],
      zip_safe=False)
