from setuptools import find_packages,setup
from typing import List

def get_req(file_path:str)->List[str]:
    requirements = []
    with open(file_path) as obj:
        requirements = obj.readlines()
        requirements = [req.replace("\n","") for req in requirements]
        if '-e .' in requirements:
            requirements.remove('-e .')

    return requirements

setup(
name="Data Science Porject",
version = '0.0.1',
author = 'Parth',
author_email='parthsaboo2004@gmail.com',
packages= find_packages(),
install_requires=get_req('requirements.txt')

)