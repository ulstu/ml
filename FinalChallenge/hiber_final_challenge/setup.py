## ! DO NOT MANUALLY INVOKE THIS setup.py, USE CATKIN INSTEAD 
from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup
# fetch values from package.xml setup_args = generate_distutils_setup(
#setup(**setup_args)
setup(
	version='0.0.1',
	scripts=[],
	packages=['hiber_final_challenge'],
	package_dir={'': 'include'} 
)
