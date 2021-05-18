from os import path
from setuptools import setup, find_packages


def parse_requirements(file):
    required_packages = []
    with open(path.join(path.dirname(__file__), file)) as req_file:
        for line in req_file:
            required_packages.append(line.strip())
    return required_packages


packages = [x for x in find_packages() if x != "test"]
setup(name='autoBOTLib',
      version='0.41',
      description="AutoBOT: Explainable AutoML for texts",
      url='https://github.com/skblaz/autobot',
#      python_requires='<3.9.0',
      author='Blaž Škrlj',
      author_email='blaz.skrlj@ijs.si',
      license='bsd-3-clause-clear',
      packages=packages,
      zip_safe=False,
      include_package_data=True,
      install_requires=parse_requirements("requirements.txt"))
