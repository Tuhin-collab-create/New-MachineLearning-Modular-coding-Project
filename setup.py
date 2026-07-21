from setuptools import setup, find_packages
from typing import List

PROJECT_NAME = "Machine Learning Project"
VERSION = "0.0.1"
DESCRIPTION = "This is our Ml Project In Modular Coding"
AUTHOR = "Tuhin Barai"
AUTHOR_EMAIL = "tuhinbarai9232475721@gmail.com"
REQUIREMENTS_FILE_NAME ="requirements.txt"
HYPEN_E_DOT = "-e ."


# open,read and operation performaed in requirement.txt

def get_requirements_list():
    with open (REQUIREMENTS_FILE_NAME) as f:
        requirement = f.readlines()
        requirement = [i.replace("\n","") for i in requirement]
        
        if HYPEN_E_DOT in requirement:
            requirement.remove(HYPEN_E_DOT)
    return requirement


setup(name=PROJECT_NAME,
    version=VERSION,
    author=AUTHOR,
    author_email = AUTHOR_EMAIL,
    packages=find_packages(),
    install_requires=get_requirements_list()
    )
    
    