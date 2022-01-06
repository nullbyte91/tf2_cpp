
#include <stdlib.h>
#include <stdio.h>
#include <string>
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/core/util/command_line_flags.h"
#include "tensorflow/core/framework/tensor_slice.h"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using tensorflow::int32;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::Tensor;
using tensorflow::tstring;
using tensorflow::SavedModelBundle;
using tensorflow::SessionOptions;
using tensorflow::RunOptions;
using tensorflow::Scope;
using tensorflow::ClientSession;

struct Prediction{
	std::unique_ptr<std::vector<std::vector<float>>> boxes;
	std::unique_ptr<std::vector<float>> scores;
	std::unique_ptr<std::vector<int>> labels;
};

class ModelLoader{
	private:
		SavedModelBundle bundle;
		SessionOptions session_options;
		RunOptions run_options;
		void make_prediction(std::vector<Tensor> &image_output, Prediction &pred);
        
	public:
		ModelLoader(string);
		//void predict(string filename, Prediction &out_pred);
        void predict(cv::Mat image, Prediction &out_pred);
        void input_processing(cv::Mat);
};


Status ReadImageFile(const string &filename, std::vector<Tensor>* out_tensors){

	//@TODO: Check if filename is valid

	using namespace ::tensorflow::ops;
	Scope root = Scope::NewRootScope();
	auto output = tensorflow::ops::ReadFile(root.WithOpName("file_reader"), filename);

	tensorflow::Output image_reader;
	const int wanted_channels = 3;
	image_reader = tensorflow::ops::DecodeJpeg(root.WithOpName("file_decoder"), output, DecodeJpeg::Channels(wanted_channels));

	auto image_unit8 = Cast(root.WithOpName("uint8_caster"), image_reader, tensorflow::DT_UINT8);
	auto image_expanded = ExpandDims(root.WithOpName("expand_dims"), image_unit8, 0);

	tensorflow::GraphDef graph;
	auto s = (root.ToGraphDef(&graph));

	if (!s.ok()){
		printf("Error in loading image from file\n");
	}
	else{
		printf("Loaded correctly!\n");
	}

	ClientSession session(root);

	auto run_status = session.Run({image_expanded}, out_tensors);
	if (!run_status.ok()){
		printf("Error in running session \n");
	}
	return Status::OK();

}

ModelLoader::ModelLoader(string path){		
	session_options.config.mutable_gpu_options()->set_allow_growth(true);	
	auto status = tensorflow::LoadSavedModel(session_options, run_options, path, {"serve"},
			&bundle);

	if (status.ok()){
		printf("Model loaded successfully...\n");
	}
	else {
		printf("Error in loading model\n");
	}

}

// void ModelLoader::predict(string filename, Prediction &out_pred){
// 	std::vector<Tensor> image_output;
// 	auto read_status = ReadImageFile(filename, &image_output);
// 	make_prediction(image_output, out_pred);
// }

void ModelLoader::input_processing(cv::Mat img){
    cv::Mat Image;
    
    int height = 224;
    int width = 224;
    int mean = 128;
    int std = 128;
    cv::Size s(height,width);
    cv::resize(img, Image, s, 0, 0, cv::INTER_CUBIC);
    int depth = img.channels();
    cv::imwrite("input.jpg", Image);

    tensorflow::Tensor input_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({1, height, width, depth}));
    auto input_tensor_mapped = input_tensor.tensor<float, 4>();
    cv::Mat Image2;
    Image.convertTo(Image2, CV_32FC1);
    Image = Image2;
    Image = Image-mean;
    Image = Image/std;
    const float * source_data = (float*) Image.data;

  // copying the data into the corresponding tensor
  for (int y = 0; y < height; ++y) {
    const float* source_row = source_data + (y * width * depth);
    for (int x = 0; x < width; ++x) {
      const float* source_pixel = source_row + (x * depth);
      for (int c = 0; c < depth; ++c) {
  const float* source_value = source_pixel + c;
  input_tensor_mapped(0, y, x, c) = *source_value;
      }
    }
  }
  std::cout << input_tensor.DebugString() << std::endl;
    const string input_node = "serving_default_input_4:0";
    // std::vector<std::pair<string, tensorflow::Tensor>> inputs_data  = {{input_node, input_tensor_mapped}};
    std::vector<string> output_nodes = {{"StatefulPartitionedCall:0",
            "StatefulPartitionedCall:1", 
            "StatefulPartitionedCall:2", 
            "StatefulPartitionedCall:3",
            "StatefulPartitionedCall:4",             
            "StatefulPartitionedCall:5",
            "StatefulPartitionedCall:6"}};
    std::vector<Tensor> predictions;
    this->bundle.GetSession()->Run({{input_node, input_tensor}}, output_nodes, {}, &predictions);
    // auto output_c = predictions[0].scalar<float>();
    std::cout << predictions[0].DebugString() << std::endl;
    for (int i = 0; i < predictions.size(); i++){
    std::string f_name = std::to_string(i) + "_name.jpg";

    tensorflow::TTypes<float>::Flat output = predictions[i].flat<float>();
    float *data_ptr = output.data();
    cv::Mat_<float> result_seg = cv::Mat_<float>::ones(224,224);
    for (int x = 0; x < 224; x++) {
    for (int y = 0; y <224; y++) {
      result_seg.at<float>(x,y) = round(*(data_ptr+224*x+y));
    }
    
    }
    //std::cout << result_seg << std::endl;
    //   cv::imwrite(f_name, result_seg);
  }
  
    // std::vector<cv::Mat> output;
    // cv::Mat mat(width, height, CV_32F);
    // std::cout << predictions[0].shape().dims() << std::endl;
    // std::cout << predictions[1].shape().dims() << std::endl;
    //std::cout << sizeof(*predictions[0]) << std::endl;
    // float* data = static_cast<float*>(predictions.data());
    // std::memcpy((void *)mat.data, camBuf , sizeof(TF_Tensor*) * NumOutputs);
    // std::cout << predictions[0] << std::endl;
    // cv::Mat rotMatrix(predictions[0].dim_size(1), predictions[0].dim_size(2), CV_32FC1, outputs[0].flat<float>().data())

    // exit(0);
    // for (int i = 0; i < predictions.size(); i++){
    //     cv::Mat mat(width, height, CV_32F);
    //     std::memcpy((void *)mat.data, predictions[0] , sizeof(predictions[0]));
    // }
}
void ModelLoader::predict(cv::Mat image, Prediction &out_pred){
	std::vector<Tensor> image_output;
    input_processing(image);
	// auto read_status = ReadImageFile(filename, &image_output);
	// make_prediction(image_output, out_pred);
}
void ModelLoader::make_prediction(std::vector<Tensor> &image_output, Prediction &out_pred){
	const string input_node = "serving_default_input_tensor:0";
	std::vector<std::pair<string, Tensor>> inputs_data  = {{input_node, image_output[0]}};
	std::vector<string> output_nodes = {{"StatefulPartitionedCall:0", //detection_anchor_indices
				"StatefulPartitionedCall:1", //detection_boxes
				"StatefulPartitionedCall:2", //detection_classes
				"StatefulPartitionedCall:3",//detection_multiclass_scores
				"StatefulPartitionedCall:4", //detection_scores                
				"StatefulPartitionedCall:5"}}; //num_detections

	
	std::vector<Tensor> predictions;
	this->bundle.GetSession()->Run(inputs_data, output_nodes, {}, &predictions);


	auto predicted_boxes = predictions[1].tensor<float, 3>();
	auto predicted_scores = predictions[4].tensor<float, 2>();
	auto predicted_labels = predictions[2].tensor<float, 2>();
	
	//inflate with predictions
	for (int i=0; i < 100; i++){
		std::vector<float> coords;
		for (int j=0; j <4 ; j++){
			coords.push_back( predicted_boxes(0, i, j));
		}
		(*out_pred.boxes).push_back(coords);
		(*out_pred.scores).push_back(predicted_scores(0, i));
		(*out_pred.labels).push_back(predicted_labels(0, i));
	}
}