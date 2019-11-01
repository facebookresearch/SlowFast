#!/bin/bash -e
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Run this script at project root by ".linter.sh" before you commit.
echo "Running isort..."
isort -y -sp .

echo "Running black..."
black -l 80 .

echo "Running flake..."
flake8 .

command -v arc > /dev/null && {
  echo "Running arc lint ..."
  arc lint
}
