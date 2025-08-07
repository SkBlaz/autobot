from pathlib import Path
from setuptools import setup, find_packages
from setuptools.command.install import install
import subprocess
import sys


def parse_requirements(file):
    """Parse requirements from requirements.txt file."""
    required_packages = []
    requirements_path = Path(__file__).parent / file
    try:
        with open(requirements_path) as req_file:
            for line in req_file:
                # Exclude any comments or empty lines
                line = line.strip()
                if line and not line.startswith("#"):
                    required_packages.append(line)
    except FileNotFoundError:
        print(f"Warning: {file} not found. Using default requirements.")
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

        try:
            print("Downloading NLTK resources...")
            for lib in ['stopwords', 'punkt_tab', 'averaged_perceptron_tagger_eng']:
                subprocess.check_call([sys.executable, "-m", "nltk.downloader", lib])
                print(f"NLTK {lib} downloaded successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Failed to download NLTK resources: {e}")
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
    install_requires=parse_requirements("requirements.txt"),
    cmdclass={
        'install': PostInstallCommand,
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research", 
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.8",
)
