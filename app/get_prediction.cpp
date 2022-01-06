#include "./saved_model_loader.h"
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

int main(int argc, char* argv[])
{
    Prediction out_pred;
	out_pred.boxes = std::unique_ptr<std::vector<std::vector<float>>>(new std::vector<std::vector<float>>());
	out_pred.scores = std::unique_ptr<std::vector<float>>(new std::vector<float>());
	out_pred.labels = std::unique_ptr<std::vector<int>>(new std::vector<int>());


    cv::VideoCapture cap("/work/assests/demo_onion_1.mp4");
    if(!cap.isOpened()){
        std::cout << "Error opening video stream or file" << std::endl;
        return -1;
    }

    std:string model_path = "/work/assests/tf/";
    ModelLoader model(model_path);

    while(1){
        cv::Mat frame;
        cap >> frame;

        if (frame.empty()){
            break;
        }
        model.predict(frame, out_pred);
        std::cout << "done" << std::endl;
    }
    cap.release();
    cv::destroyAllWindows();
    return 0;
}