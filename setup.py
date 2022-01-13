from os import path
from setuptools import setup, find_packages


def parse_requirements(file):
    required_packages = []
    with open(path.join(path.dirname(__file__), file)) as req_file:
        for line in req_file:
            required_packages.append(line.strip())
    return required_packages


long_description = """
autoBOT is an AutoML system for text classification with an emphasis on explainability.
It implements the idea of *representation evolution*, learning to combine representations
for a given task, automatically.
"""

packages = [x for x in find_packages() if x != "test"]
setup(name='autoBOTLib',
      version='1.17',
      description="AutoBOT: Explainable AutoML for texts",
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='https://github.com/skblaz/autobot',
      author='Blaž Škrlj',
      author_email='blaz.skrlj@ijs.si',
      license='bsd-3-clause-clear',
      entry_points={
          'console_scripts': ['autobot-cli = autoBOTLib.__main__:main']
      },
      packages=packages,
      zip_safe=False,
      include_package_data=True,
      install_requires=parse_requirements("requirements.txt"))
