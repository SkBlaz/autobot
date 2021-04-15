rm -rf _build
sphinx-apidoc -f -o source ../autobot;
cp source/* .;
make html;
cp -rvf _build/html/* ../docs/
