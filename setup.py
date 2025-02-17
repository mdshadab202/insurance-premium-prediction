from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> List[str]:
    """
    This function returns the list of requirements
    """
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.strip() for req in requirements]  # Remove newlines and spaces

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
        
    return requirements  # Make sure this is outside the 'with' block##

setup(
    name="data science project",
    version="0.0.1",
    author="shadab",
    author_email="shadab@tamarsoftware.in",
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)
