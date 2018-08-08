#ifndef CAFFE_OFTEN_MTCNN_LANDMARK_LOSS_LAYER_HPP_
#define CAFFE_OFTEN_MTCNN_LANDMARK_LOSS_LAYER_HPP_

#include <vector>
#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/loss_layer.hpp"

namespace caffe {

// Howto
// layer {
//  name: "loss"
//  type: "JfdaLoss"
//  bottom: "score"
//  bottom: "bbox_pred"
//  bottom: "landmark_pred"
//  bottom: "bbox_target"
//  bottom: "landmark_target"
//  bottom: "label"
//  top: "face_cls_loss"
//  top: "bbox_reg_loss"
//  top: "landmark_reg_loss"
//  top: "face_cls_pos_acc"
//  top: "face_cls_neg_acc"
//  loss_weight: 1    # face_cls_loss
//  loss_weight: 0.5  # bbox_reg_loss
//  loss_weight: 0.5  # landmark_reg_loss
//  loss_weight: 0    # no loss for neg acc
//  loss_weight: 0    # no loss for pos acc
// }

template<typename Dtype>
class OftenMtcnnLandMarkLossLayer : public LossLayer<Dtype> {
 public:
  explicit OftenMtcnnLandMarkLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param), diff_() {}

    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline int ExactNumBottomBlobs() const { return -1; }
  virtual inline int MinBottomBlobs() const { return 2; }
  virtual inline const char* type() const { return "MTCNNEuclideanLoss"; }
  /**
   * Unlike most loss layers, in the MTCNNEuclideanLossLayer we can backpropagate
   * to both inputs -- override to return true and always allow force_backward.
   */
  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return true;
  }

 protected:
  /// @copydoc MTCNNEuclideanLossLayer
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  Blob<Dtype> diff_;
  int N;
  int batch_size;
  int channel;
  int valid_label_size;
};

}  // namespace caffe

#endif  