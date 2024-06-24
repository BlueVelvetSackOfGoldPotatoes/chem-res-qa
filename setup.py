from setuptools import setup, find_packages

setup(
    name='chem-aca-q-a',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'requests',
    ],
    author='Gonçalo Hora de Carvalho',
    author_email='goncalo.horacarvalho@gmail.com',
    description='Dataset controller for chem-aca-q-a.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/BlueVelvetSackOfGoldPotatoes/chem-aca-q-a',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
