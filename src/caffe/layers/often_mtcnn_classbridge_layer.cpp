/*
* 前70的loss做回传　　label为１和-1的sample做class分类  0:负例 1：正例
*/

#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/often_mtcnn_classbridge_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void OftenMtcnnClassBridgeLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    LOG(INFO) << "Often Mtcnn Bridge.";
    if (bottom.size()!=2)
    {
        LOG(ERROR) << "Often Mtcnn Bridge Need Two Input.";
    }
    if (bottom.size()!=2)
    {
        LOG(ERROR) << "Often Mtcnn Bridge Need Two Input.";
    }
}

template <typename Dtype>
void OftenMtcnnClassBridgeLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    vector<int> top_shape = bottom[0]->shape();    
    top[0]->Reshape(top_shape);
    
    vector<int> label_shape(1);
    label_shape[0] = bottom[1]->shape(0);
    top[1]->Reshape(label_shape);    

    batch_size = bottom[0]->count(0, 1);//传递过来的shape为batch*channel*１*1
    channel = bottom[0]->count(1);
}

template <typename Dtype>
void OftenMtcnnClassBridgeLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  //前向传播　label=-1的不往前传播
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_labels = bottom[1]->cpu_data();
  //初始化top0
  Dtype* top_data = top[0]->mutable_cpu_data();
  int nCount = top[0]->count();
  caffe_set(nCount, Dtype(0), top_data);
  //初始化top1
  Dtype* top_label = top[1]->mutable_cpu_data();
  nCount = top[1]->count();
  caffe_set(nCount, Dtype(0), top_label);

  for (int i = 0; i < batch_size; i++)
  {
    if (bottom_labels[i] != -1)//传递labels 值为0和１的样本数
    {
        for (int j = 0; j < channel; j++)
        {
            top_data[i * channel + j] = bottom_data[i * channel + j ];
        }
        
    }
    top_label[i] = bottom_labels[i];

  }


}

template <typename Dtype>
void OftenMtcnnClassBridgeLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

    const Dtype* label = bottom[1]->cpu_data();

    for (int i = 0; i < batch_size; i++)
    {
        if (label[i] == -1)//传递labels 值为０和１的样本数
        {
            for (int j = 0; j < channel; j++)
            {
                bottom_diff[i * channel + j] = 0;
            }
        }
    }

    }
}
#ifdef CPU_ONLY
STUB_GPU(OftenMtcnnClassBridgeLayer);
#endif

INSTANTIATE_CLASS(OftenMtcnnClassBridgeLayer);
REGISTER_LAYER_CLASS(OftenMtcnnClassBridge);

}  // namespace caffe
