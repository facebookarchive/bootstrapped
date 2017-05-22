#!/bin/sh
python setup.py check -r -s
python -m unittest discover tests
