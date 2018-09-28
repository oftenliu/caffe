#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/often_mtcnn_softmax_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {



template <typename Dtype>
bool compGreaterByLoss(const Loss_Buffer<Dtype> &a,const Loss_Buffer<Dtype> &b)
{
    return a.loss<b.loss;
}

template <typename Dtype>
bool compGreaterByIndex(const Loss_Buffer<Dtype> &a,const Loss_Buffer<Dtype> &b)
{
    return a.index<b.index;
}

template <typename Dtype>
void OftenMtcnnSoftmaxLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);//设置loss权重为１
  LayerParameter softmax_param(this->layer_param_);
  softmax_param.set_type("Softmax");
  softmax_layer_ = LayerRegistry<Dtype>::CreateLayer(softmax_param);//使用layer_factory工厂类创建softmax_layer类对象
  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(bottom[0]);//bottom[0] 底层网络层输出数据　bottom[1]　label
  softmax_top_vec_.clear();
  softmax_top_vec_.push_back(&prob_);//softmax_bottom_vec_  softmax_top_vec_ 各push一个blob
  softmax_layer_->SetUp(softmax_bottom_vec_, softmax_top_vec_);

  has_ignore_label_ = this->layer_param_.loss_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.loss_param().ignore_label();
  }
  if (!this->layer_param_.loss_param().has_normalization() &&
      this->layer_param_.loss_param().has_normalize()) {
    normalization_ = this->layer_param_.loss_param().normalize() ?
                     LossParameter_NormalizationMode_VALID :
                     LossParameter_NormalizationMode_BATCH_SIZE;
  } else {
    normalization_ = this->layer_param_.loss_param().normalization();
  }
}

template <typename Dtype>
void OftenMtcnnSoftmaxLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  softmax_layer_->Reshape(softmax_bottom_vec_, softmax_top_vec_);
  softmax_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.softmax_param().axis());
  batch_size = bottom[0]->count(0, softmax_axis_);
  channel = bottom[0]->count(softmax_axis_);
  CHECK_EQ(batch_size, bottom[1]->count())
      << "Number of labels must match number of predictions; "
      << "e.g., if softmax axis == 1 and prediction shape is (N, C, H, W), "
      << "label count (number of labels) must be N*H*W, "
      << "with integer values in {0, 1, ..., C-1}.";
  if (top.size() >= 2) {
    // softmax output
    top[1]->ReshapeLike(*bottom[0]);
  }
  nValidLable = 0;
  loss_buffer_.resize(bottom[1]->count(0));
}

template <typename Dtype>
Dtype OftenMtcnnSoftmaxLossLayer<Dtype>::get_normalizer(
    LossParameter_NormalizationMode normalization_mode, int valid_count) {
  Dtype normalizer;
  switch (normalization_mode) {
    case LossParameter_NormalizationMode_FULL:
      normalizer = Dtype(batch_size);
      break;
    case LossParameter_NormalizationMode_VALID:
      if (valid_count == -1) {
        normalizer = Dtype(batch_size);
      } else {
        normalizer = Dtype(valid_count);
      }
      break;
    case LossParameter_NormalizationMode_BATCH_SIZE:
      normalizer = Dtype(batch_size);
      break;
    case LossParameter_NormalizationMode_NONE:
      normalizer = Dtype(1);
      break;
    default:
      LOG(FATAL) << "Unknown normalization mode: "
          << LossParameter_NormalizationMode_Name(normalization_mode);
  }
  // Some users will have no labels for some examples in order to 'turn off' a
  // particular loss in a multi-task setup. The max prevents NaNs in that case.
  return std::max(Dtype(1.0), normalizer);
}


template <typename Dtype>
Dtype OftenMtcnnSoftmaxLossLayer<Dtype>::GetTok70(vector<Loss_Buffer<Dtype>> &vecLoss,int nValid)
{

// static int logcounter = 0;

//     if(logcounter%3072 == 0)
//     {
//       LOG(INFO) << "start GetTok70============================== ";
//       for (int i = 0; i < vecLoss.size();i++)
//       {
//         LOG(INFO) << "vecLoss[" <<i << "]: loss = "<< vecLoss[i].loss<<",   index ="<<vecLoss[i].index;
//       } 
//     }


    sort(vecLoss.begin(), vecLoss.end(),compGreaterByLoss<Dtype>);//升序排列
    // if(logcounter%3072 == 0)
    // {
    //   LOG(INFO) << "loss order============================== ";
    //   for (int i = 0; i < vecLoss.size();i++)
    //   {
    //     LOG(INFO) << "vecLoss[" <<i << "]: loss = "<< vecLoss[i].loss<<",   index ="<<vecLoss[i].index;
    //   } 
    // }


    int nTop70 = nValid * 0.7;
    for (int i = 0; i < vecLoss.size() - nTop70;i++)
    {
      vecLoss[i].loss = -1;//

    }



    // if(logcounter%3072 == 0)
    // {
    //   LOG(INFO) << "loss order gettok70============================== ";
    //   for (int i = 0; i < vecLoss.size();i++)
    //   {
    //     LOG(INFO) << "vecLoss[" <<i << "]: loss = "<< vecLoss[i].loss<<",   index ="<<vecLoss[i].index;
    //   } 
    // }


    sort(vecLoss.begin(), vecLoss.end(),compGreaterByIndex<Dtype>);//升序排列

    // if(logcounter%3072 == 0)
    // {
    //   LOG(INFO) << "end GetTok70============================== ";
    //   for (int i = 0; i < vecLoss.size();i++)
    //   {
    //     LOG(INFO) << "vecLoss[" <<i << "]: loss = "<< vecLoss[i].loss<<",   index ="<<vecLoss[i].index;
    //   }
    //    logcounter = 0;
    // }
    // logcounter++;    
    return 0;
}

template <typename Dtype>
void OftenMtcnnSoftmaxLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the softmax prob values.

  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  const Dtype* prob_data = prob_.cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  nValidLable = 0;
  Dtype loss = 0;
  Dtype per_loss = 0;
  for (int i = 0; i < batch_size; ++i) {
      const int label_value = static_cast<int>(label[i]);
      if (label_value == -1 || label_value == -2) {
        loss_buffer_[i].loss = -1;
        loss_buffer_[i].index = i;
        continue;
      }
      DCHECK_GE(label_value, -2);
      DCHECK_LT(label_value, prob_.shape(softmax_axis_));
      per_loss = -log(std::max(prob_data[i * channel + label_value],Dtype(FLT_MIN)));     
      loss += per_loss;
      loss_buffer_[i].loss = per_loss;
      loss_buffer_[i].index = i;
      ++nValidLable;
  }


  top[0]->mutable_cpu_data()[0] = loss / get_normalizer(normalization_, nValidLable);
  GetTok70(loss_buffer_,nValidLable);


  //LOG(INFO) << "Loss: " << top[0]->mutable_cpu_data()[0];
  if (top.size() == 2) {
    top[1]->ShareData(prob_);
  }
  
}


template <typename Dtype>
void OftenMtcnnSoftmaxLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* prob_data = prob_.cpu_data();
    caffe_copy(prob_.count(), prob_data, bottom_diff);
    const Dtype* label = bottom[1]->cpu_data();
    int count = 0;
  

    for (int i = 0; i < batch_size; ++i) {
        const int label_value = static_cast<int>(label[i]);
        if (label_value == -1 || label_value == -2) {//如果为忽略标签，不回传梯度
          for (int c = 0; c < channel; ++c) {
            bottom_diff[i * channel + c ] = 0;
          }
        } else {//计算偏导，预测正确的bottom_diff = f(y_k) - 1，其它不变
            if (loss_buffer_[i].loss != -1){//正负样本中损失前70%的回传梯度
                bottom_diff[i * channel + label_value] -= 1;
                ++count;
            }
            else
            {
              for (int c = 0; c < channel; ++c) {
                bottom_diff[i * channel + c ] = 0;
              }
            }
        }
    }
    // Scale gradient
    Dtype loss_weight = top[0]->cpu_diff()[0] /
                        get_normalizer(normalization_, count);
    // bottom_diff = loss_weight*bottom_diff
    caffe_scal(prob_.count(), loss_weight, bottom_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU(OftenMtcnnSoftmaxLossLayer);
#endif

INSTANTIATE_CLASS(OftenMtcnnSoftmaxLossLayer);
REGISTER_LAYER_CLASS(OftenMtcnnSoftmaxLoss);

}  // namespace caffe
