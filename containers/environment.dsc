BootStrap: docker
From: ubuntu:latest

%labels

%environment
export LC_ALL=C

%post
export DEBIAN_FRONTEND=noninteractive

apt-get update
apt-get install -y libhdf5-dev graphviz locales python3-dev python3-pip curl git
apt-get clean

pip3 install alabaster==0.7.12
pip3 install attrs==20.3.0
pip3 install autoflake==1.4
pip3 install Babel==2.9.0
pip3 install certifi==2020.12.5
pip3 install chardet==4.0.0
pip3 install click==7.1.2
pip3 install cycler==0.10.0
pip3 install deap==1.3.1
pip3 install decorator==4.4.2
pip3 install dill==0.3.3
pip3 install docutils==0.16
pip3 install editdistance==0.5.3
pip3 install gensim==3.8.3
pip3 install idna==2.10
pip3 install imagesize==1.2.0
pip3 install iniconfig==1.1.1
pip3 install Jinja2==2.11.3
pip3 install joblib==1.0.1
pip3 install kiwisolver==1.3.1
pip3 install MarkupSafe==1.1.1
pip3 install matplotlib==3.3.4
pip3 install networkx==2.5
pip3 install nltk==3.5
pip3 install numpy==1.20.1
pip3 install packaging==20.9
pip3 install pandas==1.2.2
pip3 install Pillow==8.1.0
pip3 install pluggy==0.13.1
pip3 install py==1.10.0
pip3 install pyflakes==2.2.0
pip3 install Pygments==2.7.4
pip3 install pynndescent==0.5.1
pip3 install pyparsing==2.4.7
pip3 install pytest==6.2.3
pip3 install pytest-sugar==0.9.4
pip3 install python-dateutil==2.8.1
pip3 install pytz==2021.1
pip3 install regex==2020.11.13
pip3 install requests==2.25.1
pip3 install scikit-learn==0.24.1
pip3 install scipy==1.6.0
pip3 install seaborn==0.11.1
pip3 install six==1.15.0
pip3 install sklearn==0.0
pip3 install smart-open==4.1.2
pip3 install snowballstemmer==2.1.0
pip3 install Sphinx==3.4.3
pip3 install sphinx-rtd-theme==0.5.1
pip3 install sphinxcontrib-applehelp==1.0.2
pip3 install sphinxcontrib-devhelp==1.0.2
pip3 install sphinxcontrib-htmlhelp==1.0.3
pip3 install sphinxcontrib-jsmath==1.0.1
pip3 install sphinxcontrib-qthelp==1.0.3
pip3 install sphinxcontrib-serializinghtml==1.1.4
pip3 install termcolor==1.1.0
pip3 install threadpoolctl==2.1.0
pip3 install toml==0.10.2
pip3 install tqdm==4.56.0
pip3 install urllib3==1.26.3
pip3 install wget==3.2
pip3 install yapf==0.30.0

unset DEBIAN_FRONTEND