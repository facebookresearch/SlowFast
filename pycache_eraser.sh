#!/bin/sh

find . | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf
