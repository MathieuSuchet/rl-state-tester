from setuptools import setup, find_packages

long_description = ""

setup(
   name='rl-state-tester',
   version='0.1.0',
   description='',
   author='CryyStall',
   url='https://github.com/MathieuSuchet/rl-state-tester',
   packages=[package for package in find_packages() if package.startswith("rl_state_tester")],
   long_description=long_description,
   install_requires=['keyboard', 'pygame', 'numpy'],
)