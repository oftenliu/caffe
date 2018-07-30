#include <vector>

#include "caffe/layers/often_mtcnn_euclidean_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void OftenMtcnnEuclideanLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
      << "Inputs must have the same dimension.";

  diff_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void OftenMtcnnEuclideanLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    int count = bottom[0]->count();

    const Dtype* label = bottom[2]->cpu_data();
    batch_size = bottom[2]->num();
    valid_label_size = 0;
    //label
    Dtype* diff = diff_.mutable_cpu_data();
    channel = bottom[0]->channels();
    memset(diff, 0, sizeof(Dtype)*count);

    const Dtype* b0 = bottom[0]->cpu_data();
    const Dtype* b1 = bottom[1]->cpu_data();
    Dtype loss = 0;

    for (int i = 0; i < batch_size; ++i){//batchsize
        if (label[i] != 0){
            caffe_sub(//y[i] = a[i] + - * \ b[i]
                channel,
                b0 + i * channel,
                b1 + i * channel,
                diff + i * channel);
            Dtype dot = caffe_cpu_dot(channel, diff + i * channel, diff + i * channel);
            loss += dot;
            valid_label_size++;
        }
    }
    top[0]->mutable_cpu_data()[0] = loss / valid_label_size / Dtype(2);
}

template <typename Dtype>
void OftenMtcnnEuclideanLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

    const Dtype* label = bottom[2]->cpu_data();
    for (int i = 0; i < 2; ++i) {
        if (propagate_down[i]) {
            memset(bottom[i]->mutable_cpu_diff(), 0, sizeof(Dtype)*bottom[i]->count());

            const Dtype sign = (i == 0) ? 1 : -1;
            const Dtype alpha = sign * top[0]->cpu_diff()[0] / valid_label_size;

            for (int j = 0; j < batch_size; ++j){
                if (label[j] != 0){
                    caffe_cpu_axpby(
                        channel,							// count
                        alpha,                              // alpha
                        diff_.cpu_data() + channel * j,                   // a
                        Dtype(0),                           // beta
                        bottom[i]->mutable_cpu_diff() + channel * j);  // b
                }
            }
        }
    }


}

#ifdef CPU_ONLY
STUB_GPU(OftenMtcnnEuclideanLossLayer);
#endif

INSTANTIATE_CLASS(OftenMtcnnEuclideanLossLayer);
REGISTER_LAYER_CLASS(OftenMtcnnEuclideanLoss);

}  // namespace caffe
