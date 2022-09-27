#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <iostream>

using namespace cv;
using namespace std;


int main(int argc, char **argv){

    //Load Image
    string path = "../resources/lenna.png";
    Mat img = imread(path);

    // Check if image loaded successfully
    if (img.empty()){
        perror("Image was not loaded");
        exit(-1);
    }

    //Load haarcascade
    CascadeClassifier faceCascade;
    faceCascade.load("../resources/haarcascade_frontalface_default.xml");

    if (faceCascade.empty()){
        perror("XML File not loaded");
        exit(-1);
    }

    Mat img_gray;
    cvtColor(img, img_gray, COLOR_BGR2GRAY);

    vector<Rect> faces;
    faceCascade.detectMultiScale(img_gray, faces, 1.1, 10);

    for (int i = 0; i < faces.size(); i++){
        Point top_left = faces[i].tl();
        Point bot_right = faces[i].br();
        rectangle(img, top_left, bot_right, Scalar(0,0,0), 3);
        Mat faceROI = img(faces[i]);
    }

    imshow("Image", img);

    if (waitKey(0) == 'q'){
        return 0;
    }
    
    return 0;
}