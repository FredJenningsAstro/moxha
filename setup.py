from setuptools import setup, find_packages


setup(name = 'moxha',
      version = '0.1',
      description = 'X-ray Analysis Toolkit',
      url = '',
      author = 'Fred Jennings',
      author_email = 'fred.jennings@ed.ac.uk',
      license = 'GPLv3',
      packages = find_packages(),
      include_package_data=True,
      zip_safe = False, install_requires=['numpy', 'scipy', 'h5py', 'pyxsim', 'soxs'])