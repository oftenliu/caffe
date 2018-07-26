#!/usr/bin/env sh
cd /home/ulsee/often/caffe/examples/mtcnn/
rm -rf pnet_db
/home/ulsee/often/caffe/build/tools/convert_mtcnn_imageset / /home/ulsee/lqc/often-mtcnn/tmp/data/pnet/train_pnet.txt /home/ulsee/often/caffe/examples/mtcnn/pnet_db --shuffle=true

