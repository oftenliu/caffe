#!/usr/bin/env sh 
/home/ulsee/often/caffe/build/tools/caffe train  \
                                          --solver=/home/ulsee/often/caffe/examples/mtcnn/solver.prototxt  \
                                          --weights=/home/ulsee/often/caffe/models/solver_iter_30000.caffemodel 

