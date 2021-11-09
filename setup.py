from setuptools import setup, find_packages

setup(name='cairo_simulator',
      version='0.1',
      description='Package containing interfaces and algorithms for simulating robots and planning.',
      url='https://github.com/cairo-robotics/cairo_simulator',
      author='Carl Mueller',
      author_email='carl.mueller@colorado.edu',
      license='',
      package_dir={'': 'src'},
      packages=find_packages(where='src'),
      install_requires=["pybullet==3.1.8", "scikit-learn==0.23.1", "python-igraph==0.9.6", "pyquaternion==0.9.5", "ikpy==3.0.1", "numpy==1.19.2", "networkx==2.6.3"],
      include_package_data=True)