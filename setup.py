from setuptools import setup, find_packages

setup(
    name='predictive_maintenance',
    version='0.1.0',
    author='Your Name',
    description='An end-to-end machine learning pipeline for predictive maintenance using sensor data',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/Opikadash/predictive-maintenance-model',
    packages=find_packages(include=['src', 'src.*']),
    install_requires=[
        'pandas',
        'scikit-learn',
        'joblib',
        'streamlit',
        'Flask',
        'matplotlib'
    ],
    include_package_data=True,
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)
