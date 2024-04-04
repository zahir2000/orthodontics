from setuptools import setup, find_packages

setup(
    name='orthodontics',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
        # Add other dependencies required by your package
    ],
    author='Zahiriddin',
    author_email='contact@zahiriddin.com',
    description='A package for Orthodontics model training',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/zahir2000/orthodontics',
)