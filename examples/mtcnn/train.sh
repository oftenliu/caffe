#!/usr/bin/env sh
set -e

./build/tools/caffe train --solver=examples/mtcnn/solver.prototxt $@