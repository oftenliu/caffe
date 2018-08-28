//制作训练mtcnn网络数据

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "boost/scoped_ptr.hpp"
#include "gflags/gflags.h"
#include "glog/logging.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/format.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"

using namespace caffe;  // NOLINT(build/namespaces)
using std::pair;
using boost::scoped_ptr;

DEFINE_bool(gray, false,
    "When this option is on, treat images as grayscale ones");
DEFINE_bool(shuffle, false,
    "Randomly shuffle the order of images and their labels");
DEFINE_string(backend, "lmdb",
        "The backend {lmdb, leveldb} for storing the result");
DEFINE_int32(resize_width, 0, "Width images are resized to");
DEFINE_int32(resize_height, 0, "Height images are resized to");
DEFINE_bool(check_size, false,
    "When this option is on, check that all the datum have the same size");
DEFINE_bool(encoded, false,
    "When this option is on, the encoded image will be save in datum");
DEFINE_string(encode_type, "",
    "Optional: What type should we encode the image as ('png','jpg',...).");




void split(const string& s, char delim,std::vector<string>& v) {
    auto i = 0;
    auto pos = s.find(delim);

    while (pos != string::npos) {
      v.push_back(s.substr(i, pos-i));
      i = ++pos;
      pos = s.find(delim, pos);
    }
    if (pos == string::npos)
      v.push_back(s.substr(i, s.length()));
}


int main(int argc, char** argv) {
    LOG(INFO) << "0000";
#ifdef USE_OPENCV
  ::google::InitGoogleLogging(argv[0]);
  // Print output to stderr (while still logging)
  FLAGS_alsologtostderr = 1;

#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif

  gflags::SetUsageMessage("Convert a set of images to the leveldb/lmdb\n"
        "format used as input for Caffe.\n"
        "Usage:\n"
        "    convert_imageset [FLAGS] ROOTFOLDER/ LISTFILE DB_NAME\n"
        "The ImageNet dataset for the training demo is at\n"
        "    http://www.image-net.org/download-images\n");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (argc < 4) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/convert_imageset");
    return 1;
  }

  const bool is_color = !FLAGS_gray;
  const bool check_size = FLAGS_check_size;
  const bool encoded = FLAGS_encoded;
  const string encode_type = FLAGS_encode_type;

  std::ifstream infile(argv[2]);
  std::vector<std::pair<std::string, std::vector<float>> > lines;
  std::string line;
  size_t pos;
  std::string strLabels;
  LOG(INFO) << "0000";
  while (std::getline(infile, line)) {
    pos = line.find_first_of(' ');
    strLabels = line.substr(pos + 1).c_str();
    std::vector<std::string> strLabelVec;
	  split(strLabels,' ',strLabelVec);
    std::vector<float> fLabelVec(15,0);
    fLabelVec[0] = atoi(strLabelVec[0].c_str());
    if (strLabelVec.size() == 5)
    {
      for (int i = 1; i< strLabelVec.size(); i++)
      {
        fLabelVec[i] = atof(strLabelVec[i].c_str());
      }
    }
    if (strLabelVec.size() == 11)
    {
      for (int i = 1; i< strLabelVec.size(); i++)
      {
        fLabelVec[i+4] = atof(strLabelVec[i].c_str());
      }
    }
    // printf("\n========1111111111111111   \n");
    // for (int i = 0; i < fLabelVec.size();i++)
    // {
      
    //   printf("%f ",fLabelVec[i]);
      
    // }
    // printf("\n  ========2222222222222222\n");
    lines.push_back(std::make_pair(line.substr(0, pos), fLabelVec));

  }
  LOG(INFO) << "1111";
  if (FLAGS_shuffle) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    shuffle(lines.begin(), lines.end());
  }
  LOG(INFO) << "A total of " << lines.size() << " images.";

  if (encode_type.size() && !encoded)
    LOG(INFO) << "encode_type specified, assuming encoded=true.";

  int resize_height = std::max<int>(0, FLAGS_resize_height);
  int resize_width = std::max<int>(0, FLAGS_resize_width);

  // Create new DB
  scoped_ptr<db::DB> db(db::GetDB(FLAGS_backend));
  db->Open(argv[3], db::NEW);
  scoped_ptr<db::Transaction> txn(db->NewTransaction());
    LOG(INFO) << "23545453545."; 
  // Storing to db
  std::string root_folder(argv[1]);
  MTCNNDatum mtcnndatum;
  int count = 0;
  int data_size = 0;
  bool data_size_initialized = false;

  for (int line_id = 0; line_id < lines.size(); ++line_id) {
    bool status;
    std::string enc = encode_type;
    if (encoded && !enc.size()) {
      // Guess the encoding type from the file name
      string fn = lines[line_id].first;
      size_t p = fn.rfind('.');
      if ( p == fn.npos )
        LOG(WARNING) << "Failed to guess the encoding of '" << fn << "'";
      enc = fn.substr(p+1);
      std::transform(enc.begin(), enc.end(), enc.begin(), ::tolower);
    }
    status = ReadImageToMTCNNDatum(root_folder + lines[line_id].first,
			lines[line_id].second, resize_height, resize_width, is_color,
			enc, &mtcnndatum);
    if (status == false) continue;
    if (check_size) {
      if (!data_size_initialized) {
        data_size = mtcnndatum.datum().channels() * mtcnndatum.datum().height() * mtcnndatum.datum().width();
        data_size_initialized = true;
      } else {
        const std::string& data = mtcnndatum.datum().data();
        CHECK_EQ(data.size(), data_size) << "Incorrect data field size "
            << data.size();
      }
    }
    // sequential
    string key_str = caffe::format_int(line_id, 8) + "_" + lines[line_id].first;

    // Put in db
    string out;
    CHECK(mtcnndatum.SerializeToString(&out));
    txn->Put(key_str, out);

    if (++count % 1000 == 0) {
      // Commit db
      txn->Commit();
      txn.reset(db->NewTransaction());
      LOG(INFO) << "Processed " << count << " files.";
    }
  }
  // write the last batch
  if (count % 1000 != 0) {
    txn->Commit();
    LOG(INFO) << "Processed " << count << " files.";
  }
#else
  LOG(FATAL) << "This tool requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
  return 0;
}
