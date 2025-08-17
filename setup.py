from pathlib import Path
from setuptools import setup, find_packages


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
