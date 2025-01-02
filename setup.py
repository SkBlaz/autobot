from os import path
from setuptools import setup, find_packages
from setuptools.command.install import install
import subprocess
import sys

def parse_requirements(file):
    required_packages = []
    with open(path.join(path.dirname(__file__), file)) as req_file:
        for line in req_file:
            # Exclude any comments or empty lines
            line = line.strip()
            if line and not line.startswith("#"):
                required_packages.append(line)
    return required_packages

class PostInstallCommand(install):
    """Post-installation for downloading NLTK resources."""
    def run(self):
        install.run(self)
        
        try:
            import nltk
        except ImportError:
            print("NLTK is not installed. Installing NLTK...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "nltk"])
            import nltk        
        try:
            print("Downloading NLTK 'stopwords' resource...")
            for lib in ['stopwords', 'punkt_tab', 'averaged_perceptron_tagger_eng']:
                subprocess.check_call([sys.executable, "-m", "nltk.downloader", lib])
                print(f"NLTK {lib} downloaded successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Failed to download NLTK 'stopwords': {e}")
            sys.exit(1)  # Exit with error code

long_description = """
autoBOT is an AutoML system for text classification with an emphasis on explainability.
It implements the idea of *representation evolution*, learning to combine representations
for a given task, automatically.
"""

packages = [x for x in find_packages() if x != "test"]

setup(
    name='autoBOTLib',
    version='1.19',
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
    install_requires=parse_requirements("requirements.txt")
)
