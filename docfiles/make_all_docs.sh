rm -rf _build
sphinx-apidoc -f -o source ../autobot;
cp source/* .;
make html;
sphinx-apidoc -f -o source ../autoBOTLib;
cp source/* .;
cp -rvf _build/html/* ../docs/;
