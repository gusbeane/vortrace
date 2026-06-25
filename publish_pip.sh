#!/bin/bash

source venv/bin/activate
python -m pip install --upgrade build twine
rm -rf dist/
python -m build
python -m twine check dist/*
python -m twine upload dist/*
