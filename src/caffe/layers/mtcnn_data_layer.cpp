#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif  // USE_OPENCV
#include <stdint.h>

#include <boost/thread.hpp>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/mtcnn_data_layer.hpp"
#include "caffe/util/benchmark.hpp"

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/blocking_queue.hpp"


namespace caffe {

template <typename Dtype>
MTCNNDataLayer<Dtype>::MTCNNDataLayer(const LayerParameter& param)
: BasePrefetchingDataLayer<Dtype>(param),
    offset_() {
    db_.reset(db::GetDB(param.data_param().backend()));
    db_->Open(param.data_param().source(), db::READ);
    cursor_.reset(db_->NewCursor());
}

template <typename Dtype>
MTCNNDataLayer<Dtype>::~MTCNNDataLayer() {
    this->StopInternalThread();
}

template <typename Dtype>
void MTCNNDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

    this->output_labels_ = top.size() > 1;
    this->output_roi_ = top.size() > 2;
    this->output_pts_ = top.size() > 3;    
    // Read a data point, and use it to initialize the top blob.
    Datum datum;//Datum用来从LMDB/LEVELDB 中读取数据
    datum.ParseFromString(cursor_->value());



    int batch_size = this->layer_param_.data_param().batch_size();
  
    //data, label, roi, pts
    // Use data_transformer to infer the expected blob shape from datum.
    vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);
    //��֤ת������ԭͼ�ߴ�
    this->transformed_data_.Reshape(top_shape);


    // Reshape top[0] and prefetch_data according to the batch_size.

    top_shape[0] = batch_size;
    //top_shape[2] = resize_height;
    //top_shape[3] = resize_width;

    //data
    top[0]->Reshape(top_shape);
    for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
        this->prefetch_[i].data_.Reshape(top_shape);
    }
    LOG(INFO) << "output data size: " << top[0]->num() << ","
        << top[0]->channels() << "," << top[0]->height() << ","
        << top[0]->width();

    // label
    vector<int> label_shape(2);
    label_shape[0] = batch_size;
    label_shape[1] = datum.datum().labels_size() == 0 ? 1 : datum.datum().labels_size();
    top[1]->Reshape(label_shape);
    for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
        this->prefetch_[i].label_.Reshape(label_shape);
    }

    //roi
    vector<int> roi_shape(2);
    roi_shape[0] = batch_size;
    roi_shape[1] = 4;
    top[2]->Reshape(roi_shape);
    for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
        this->prefetch_[i].roi_.Reshape(roi_shape);
    }

    //pts
    if (output_pts_){
        vector<int> pts_shape(2);
        pts_shape[0] = batch_size;
        pts_shape[1] = datum.pts_size();
        top[3]->Reshape(pts_shape);
        for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
            this->prefetch_[i].pts_.Reshape(pts_shape);
        }
    }
}

template <typename Dtype>
void MTCNNDataLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    Batch<Dtype>* batch = prefetch_full_.pop("Data layer prefetch queue empty");
    // Reshape to loaded data.
    top[0]->ReshapeLike(batch->data_);
    // Copy the data
    caffe_copy(batch->data_.count(), batch->data_.cpu_data(),
        top[0]->mutable_cpu_data());
    DLOG(INFO) << "Prefetch copied";

    //label
    top[1]->ReshapeLike(batch->label_);
    caffe_copy(batch->label_.count(), batch->label_.cpu_data(),
        top[1]->mutable_cpu_data());

    //roi
    top[2]->ReshapeLike(batch->roi_);
    caffe_copy(batch->roi_.count(), batch->roi_.cpu_data(),
        top[2]->mutable_cpu_data());

    if (output_pts_){
        //pts
        top[3]->ReshapeLike(batch->pts_);
        caffe_copy(batch->pts_.count(), batch->pts_.cpu_data(),
            top[3]->mutable_cpu_data());
    }

    prefetch_free_.push(batch);
}


template<typename Dtype>
void MTCNNDataLayer<Dtype>::Next() {
  cursor_->Next();
  if (!cursor_->valid()) {
    LOG_IF(INFO, Caffe::root_solver())
        << "Restarting data prefetching from start.";
    cursor_->SeekToFirst();
  }
  offset_++;
}



// This function is called on prefetch thread
template<typename Dtype>
void MTCNNDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
    CPUTimer batch_timer;
    batch_timer.Start();
    double read_time = 0;
    double trans_time = 0;
    CPUTimer timer;
    CHECK(batch->data_.count());
    CHECK(this->transformed_data_.count());

    // Reshape according to the first datum of each batch
    // on single input batches allows for inputs of varying dimension.
    const int batch_size = this->layer_param_.data_param().batch_size();

    MTCNNDatum datum;//Datum用来从LMDB/LEVELDB 中读取数据
    datum.ParseFromString(cursor_->value());
    // Use data_transformer to infer the expected blob shape from datum.
    vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);
    this->transformed_data_.Reshape(top_shape);
    // Reshape batch according to the batch_size.
    top_shape[0] = batch_size;
    batch->data_.Reshape(top_shape);

    //Ϊ��ͼ׼��һ�����������û�������ͼ����ƽ��ŵģ���Ҫ�任�ɷ�ͨ��
    Mat buffer(raw_image_shape_[2], raw_image_shape_[3], CV_32FC(raw_image_shape_[1]));
    Dtype* top_data = batch->data_.mutable_cpu_data();
    Dtype* top_label = batch->label_.mutable_cpu_data();
    Dtype* top_roi = batch->roi_.mutable_cpu_data();

    //Dtype* top_pts = output_pts_ ? batch->pts_.mutable_cpu_data() : 0;
    for (int item_id = 0; item_id < raw_batch_size; ++item_id) {
        timer.Start();

        // get a datum
        datum.ParseFromString(cursor_->value());
        const Datum& data = datum.datum();
        //printf("****************datum.pts_size() = %d\n", datum.pts_size());  

        read_time += timer.MicroSeconds();
        timer.Start();
        //获取data数据
        int offset = batch->data_.offset(item_id);
        this->transformed_data_.set_cpu_data(top_data + offset);
        this->data_transformer_->Transform(datum, &(this->transformed_data_));
        // Copy label.
        if (this->output_labels_) {
            top_label[item_id] = datum.label();
        }
        top_roi[item_id+0] = datum.rois.xmin();
        top_roi[item_id+1] = datum.rois.ymin();
        top_roi[item_id+2] = datum.rois.xmax() - datum.rois.xmin() + 1;
        top_roi[item_id+3] = datum.rois.ymax() - datum.rois.ymin() + 1);
        trans_time += timer.MicroSeconds();
        Next();

    }

    timer.Stop();
    batch_timer.Stop();
    DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
    DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
    DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}


INSTANTIATE_CLASS(MTCNNDataLayer);
REGISTER_LAYER_CLASS(MTCNNData);

}  // namespace caffe
