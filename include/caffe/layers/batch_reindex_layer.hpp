#ifndef CAFFE_BATCHREINDEX_LAYER_HPP_
#define CAFFE_BATCHREINDEX_LAYER_HPP_

#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Index int_tpo the input blob along its first axis.
 *
 * This layer can be used to select, reorder, and even replicate examples in a
 * batch.  The second blob is cast to int_tp and treated as an index int_tpo the
 * first axis of the first blob.
 */
template<typename Dtype, typename MItype, typename MOtype>
class BatchReindexLayer : public Layer<Dtype, MItype, MOtype> {
 public:
  explicit BatchReindexLayer(const LayerParameter& param)
      : Layer<Dtype, MItype, MOtype>(param) {}
  virtual void Reshape(const vector<Blob<MItype>*>& bottom,
      const vector<Blob<MOtype>*>& top);

  virtual inline const char* type() const { return "BatchReindex"; }
  virtual inline int_tp ExactNumBottomBlobs() const { return 2; }
  virtual inline int_tp ExactNumTopBlobs() const { return 1; }

 protected:
  /**
   * @param bottom input Blob vector (length 2+)
   *   -# @f$ (n \times ...) @f$
   *      the inputs @f$ x_1 @f$
   *   -# @f$ (m) @f$
   *      the inputs @f$ x_2 @f$
   * @param top output Blob vector (length 1)
   *   -# @f$ (m \times ...) @f$:
   *      the reindexed array @f$
   *        Y = x_1[x_2]
   *      @f$
   */
  virtual void Forward_cpu(const vector<Blob<MItype>*>& bottom,
      const vector<Blob<MOtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<MItype>*>& bottom,
      const vector<Blob<MOtype>*>& top);

  /**
   * @brief Computes the error gradient w.r.t. the reordered input.
   *
   * @param top output Blob vector (length 1), providing the error gradient
   *        with respect to the outputs
   *   -# @f$ (m \times ...) @f$:
   *      containing error gradients @f$ \frac{\partial E}{\partial Y} @f$
   *      with respect to concatenated outputs @f$ Y @f$
   * @param propagate_down see Layer::Backward.
   * @param bottom input Blob vector (length 2):
   *   - @f$ \frac{\partial E}{\partial Y} @f$ is de-indexed (summing where
   *     required) back to the input x_1
   *   - This layer cannot backprop to x_2, i.e. propagate_down[1] must be
   *     false.
   */
  virtual void Backward_cpu(const vector<Blob<MOtype>*>& top,
      const vector<bool>& propagate_down,
      const vector<Blob<MItype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<MOtype>*>& top,
      const vector<bool>& propagate_down,
      const vector<Blob<MItype>*>& bottom);

 private:
  void GenerateProgram();
  struct pair_sort_first {
    bool operator()(const pair<int_tp, int_tp> &left,
                    const pair<int_tp, int_tp> &right) {
      return left.first < right.first;
    }
  };
  void check_batch_reindex(int_tp initial_num, int_tp final_num,
                           const Dtype* ridx_data);
};

}  // namespace caffe

#endif  // CAFFE_BATCHREINDEX_LAYER_HPP_
