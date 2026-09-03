from setuptools import setup, find_packages

# See setup details from https://python-packaging.readthedocs.io/en/latest/minimal.html
# pip install -e .
setup(name='Stochastic Cognitive Dynamics',
      version='0.1',
      description='Drift diffusion model and Qauntum Probability models used in Psychology',
      url='',
      author='Ritesh K Malaiya',
      author_email='ritesh.malaiya@gmail.com',
      license='MIT',
      # packages=['cme'] shipped cme/__init__.py and nothing else - every subpackage
      # was dropped on a non-editable install. Editable installs hid it by putting the
      # repo root on sys.path.
      packages=find_packages(include=['cme', 'cme.*']),
      python_requires='>=3.12',
      # arviz>=1.2 is the refactored 1.x line; fit_model uses dict_to_dataset's
      # sample_dims, which only exists there. It also requires Python >=3.12, which is
      # what python_requires above is really tracking.
      # Left unpinned other than arviz: these are the packages cme actually imports,
      # and no version floor has been established for them yet.
      install_requires=[
          'arviz[h5netcdf,matplotlib]>=1.2,<2',
          'attrs',
          'jax',
          'joblib',
          'matplotlib',
          'numpy',
          'numpyro',
          'optax',
          'pandas',
          'polars',
          'pymc',
          'scipy',
          'seaborn',
          'xarray',
      ],
      zip_safe=False)
