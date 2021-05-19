rm -rf _build;rm -rf source/*
make html;
sphinx-apidoc -f -o source ../autoBOTLib;
cp source/* .;
cp -rvf _build/html/* ../docs/;
