#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(){

    //Load Video
    string path = "../resources/videoIn.mp4";
    VideoCapture cap(path);

    // Check if camera opened successfully
    if (!cap.isOpened()){
        perror("Error opening video stream or file");
        exit(-1);
    }

    //Load haarcascade
    CascadeClassifier faceCascade;
    faceCascade.load("../resources/haarcascade_frontalface_default.xml");

    if (faceCascade.empty()){
        perror("XML File not loaded");
        exit(-1);
    }
    
    while (true){
        Mat img;
        cap >> img;
        if (img.empty()){
            cout << "No frame captured from video" << endl;
            break;
        }

        resize(img, img, Size(640,480));

        Mat img_gray;
        cvtColor(img, img_gray, COLOR_BGR2GRAY);

        vector<Rect> faces;
        faceCascade.detectMultiScale(img_gray, faces, 1.1, 10);

        for (int i = 0; i < faces.size(); i++){
            rectangle(img, faces[i].tl(), faces[i].br(), Scalar(0,0,0), 3);
            Mat faceROI = img_gray(faces[i]);
        }

        imshow("Video Face Detection", img);

        if (waitKey(1) == 'q'){
            break;
        }
    }

    cap.release();

    destroyAllWindows();
    
    return 0;
}