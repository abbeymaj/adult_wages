# Importing packages
from typing import List
from setuptools import setup, find_packages

# Setting hyphen e dot as a global variable
HYPHEN_E_DOT = '-e .'

# Creating a function to fetch all packages from requirements.txt
def get_requirements(filepath:str)->List[str]:
    '''
    This function fetches all packages from requirements.txt.
    =============================================================================================
    ----------------
    Parameters:
    ----------------
    filepath : str -> This is the filepath of the requirements.txt file. 
    This must be a string.
    
    ----------------
    Returns:
    ----------------
    List of packages : str -> This is a list of the packages from the requirements.txt file.
    =============================================================================================
    '''
    requirements = []
    with open(filepath) as f_obj:
        requirements = f_obj.readlines()
        
        # Removing line breaks if any
        requirements = [req.replace("\n", "") for req in requirements]
        
        # Removing -e . if present in requirements
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
            
    return requirements

# Using the setup function from setuptools to install packages from requirements.txt
setup(
    name='adult_wages_prediction',
    author='Abhijit Majumdar',
    version='0.0.1',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)