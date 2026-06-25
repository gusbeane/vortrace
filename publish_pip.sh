#!/bin/bash

source venv/bin/activate
python -m pip install --upgrade build twine
rm -rf dist/
python -m build --sdist

# download artifacts from build
# gh run download RUN_ID --dir dist/
mv dist/cibw-wheels*/*.whl dist/
rm -r dist/cibw-wheels*

python -m twine check dist/*
python -m twine upload dist/*
