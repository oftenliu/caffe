#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif  // USE_OPENCV
#include <stdint.h>

#include <boost/thread.hpp>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/often_mtcnn_data_layer.hpp"
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
OftenMtcnnDataLayer<Dtype>::OftenMtcnnDataLayer(const LayerParameter& param)
: BasePrefetchingDataLayer<Dtype>(param),
    offset_() {
    db_.reset(db::GetDB(param.data_param().backend()));
    db_->Open(param.data_param().source(), db::READ);
    cursor_.reset(db_->NewCursor());
}

template <typename Dtype>
OftenMtcnnDataLayer<Dtype>::~OftenMtcnnDataLayer() {
    this->StopInternalThread();
}

template <typename Dtype>
void OftenMtcnnDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

    this->output_labels_ = top.size() > 1;
   
    // Read a data point, and use it to initialize the top blob.
    MTCNNDatum datum;//Datum用来从LMDB/LEVELDB 中读取数据
    datum.ParseFromString(cursor_->value());
    int batch_size = this->layer_param_.data_param().batch_size();
  
    //data, label, roi, pts
    // Use data_transformer to infer the expected blob shape from datum.
    vector<int> top_shape = this->data_transformer_->InferBlobShape(datum.datum());
    //��֤ת������ԭͼ�ߴ�
    this->transformed_data_.Reshape(top_shape);


    // Reshape top[0] and prefetch_data according to the batch_size.

    top_shape[0] = batch_size;
    //top_shape[2] = resize_height;
    //top_shape[3] = resize_width;

    //data
    top[0]->Reshape(top_shape);
    for (int i = 0; i < this->prefetch_.size(); ++i) {
        this->prefetch_[i]->data_.Reshape(top_shape);
    }
    LOG(INFO) << "output data size: " << top[0]->num() << ","
        << top[0]->channels() << "," << top[0]->height() << ","
        << top[0]->width();

    // label
    vector<int> label_shape(1);
    label_shape[0] = batch_size;
    top[1]->Reshape(label_shape);
    for (int i = 0; i < this->prefetch_.size(); ++i) {
        this->prefetch_[i]->label_.Reshape(label_shape);
    }

    //roi
    vector<int> roi_shape(2);
    roi_shape[0] = batch_size;
    roi_shape[1] = 4;
    top[2]->Reshape(roi_shape);
    for (int i = 0; i < this->prefetch_.size(); ++i) {
        this->prefetch_[i]->roi_.Reshape(roi_shape);
    }

    //landmark
    // vector<int> landmark_shape(2);
    // landmark_shape[0] = batch_size;
    // landmark_shape[1] = 10;
    // top[3]->Reshape(landmark_shape);
    // for (int i = 0; i < this->prefetch_.size(); ++i) {
    //     this->prefetch_[i]->landmark_.Reshape(landmark_shape);
    // }    

}

template <typename Dtype>
void OftenMtcnnDataLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    Batch<Dtype>* batch = this->prefetch_full_.pop("Data layer prefetch queue empty");
    // Reshape to loaded data.
    top[0]->ReshapeLike(batch->data_);
    // Copy the data
    caffe_copy(batch->data_.count(), batch->data_.cpu_data(),
        top[0]->mutable_cpu_data());
    //DLOG(INFO) << "Prefetch copied";

    //label
    top[1]->ReshapeLike(batch->label_);
    caffe_copy(batch->label_.count(), batch->label_.cpu_data(),
        top[1]->mutable_cpu_data());

    //roi
    top[2]->ReshapeLike(batch->roi_);
    caffe_copy(batch->roi_.count(), batch->roi_.cpu_data(),
        top[2]->mutable_cpu_data());

    //landmark
    //top[3]->ReshapeLike(batch->landmark_);
    //caffe_copy(batch->landmark_.count(), batch->landmark_.cpu_data(),
        //top[3]->mutable_cpu_data());    

    this->prefetch_free_.push(batch);
}


template<typename Dtype>
void OftenMtcnnDataLayer<Dtype>::Next() {
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
void OftenMtcnnDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
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
    vector<int> top_shape = this->data_transformer_->InferBlobShape(datum.datum());
    this->transformed_data_.Reshape(top_shape);
    // Reshape batch according to the batch_size.
    top_shape[0] = batch_size;
    batch->data_.Reshape(top_shape);

    //Ϊ��ͼ׼��һ�����������û�������ͼ����ƽ��ŵģ���Ҫ�任�ɷ�ͨ��
    
    Dtype* top_data = batch->data_.mutable_cpu_data();
    Dtype* top_label = batch->label_.mutable_cpu_data();
    Dtype* top_roi = batch->roi_.mutable_cpu_data();

    //landmark
    //Dtype* top_landmark = batch->landmark_.mutable_cpu_data();
    
    for (int item_id = 0; item_id < batch_size; ++item_id) {
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
        this->data_transformer_->Transform(datum.datum(), &(this->transformed_data_));
        // Copy label.
        if (this->output_labels_) {
            top_label[item_id] = data.label();
        }
        top_roi[item_id+0] = datum.rois().xmin();
        top_roi[item_id+1] = datum.rois().ymin();
        top_roi[item_id+2] = datum.rois().xmax() - datum.rois().xmin() + 1;
        top_roi[item_id+3] = datum.rois().ymax() - datum.rois().ymin() + 1;

        //landmark
        // top_landmark[item_id + 0] = datum.landmark().xlefteye();
        // top_landmark[item_id + 1] = datum.landmark().ylefteye();
        // top_landmark[item_id + 2] = datum.landmark().xrighteye();
        // top_landmark[item_id + 3] = datum.landmark().yrighteye();
        // top_landmark[item_id + 4] = datum.landmark().xnose();
        // top_landmark[item_id + 5] = datum.landmark().ynose();
        // top_landmark[item_id + 6] = datum.landmark().xleftmouth();
        // top_landmark[item_id + 7] = datum.landmark().yleftmouth();
        // top_landmark[item_id + 8] = datum.landmark().xrightmouth();
        // top_landmark[item_id + 9] = datum.landmark().yrightmouth();


        // DLOG(INFO) << "top_roi xmin: " << top_roi[item_id+0];
        // DLOG(INFO) << "top_roi ymin: " << top_roi[item_id+1];
        // DLOG(INFO) << "top_roi xmax: " << top_roi[item_id+2];
        // DLOG(INFO) << "top_roi ymax: " << top_roi[item_id+3];
        
        trans_time += timer.MicroSeconds();
        Next();

    }

    timer.Stop();
    batch_timer.Stop();
    //DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
    //DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
    //DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}


INSTANTIATE_CLASS(OftenMtcnnDataLayer);
REGISTER_LAYER_CLASS(OftenMtcnnData);

}  // namespace caffe
