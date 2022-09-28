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

        int r = 10;
        int area = pow(2*r+1, 2);
        
        Rect R = faces[i];
        R.x -= r;
        R.y -= r;
        R.width += 2*r;
        R.height += 2*r;

        rectangle(img, faces[i].tl(), faces[i].br(), Scalar(0, 255,0), 3);

        Point top_left = R.tl();
        Point bot_right = R.br();

        int w = bot_right.x - top_left.x;
        int h = bot_right.y - top_left.y;
        
        rectangle(img, top_left, bot_right, Scalar(0,0,0), 3);

        img.convertTo(img, CV_32SC3);
        Mat faceROI = img(R);

        faceROI.convertTo(faceROI, CV_32SC3);

        Mat table;
        table = Mat::zeros(faceROI.size(), CV_32SC3);

        table.at<Vec3i>(0,0) = faceROI.at<Vec3i>(0,0);
        for (int x = 1; x < w; x++){
            table.at<Vec3i>(0,x) = faceROI.at<Vec3i>(0,x) + table.at<Vec3i>(0,x-1);
        }

        for (int y = 1; y < h; y++){
            table.at<Vec3i>(y,0) = faceROI.at<Vec3i>(y,0) + table.at<Vec3i>(y-1,0);
        }
        
        for (int y = 1; y < h-1; y++){
            for (int x = 1; x < w-1; x++){
                table.at<Vec3i>(y,x) = faceROI.at<Vec3i>(y,x) + table.at<Vec3i>(y-1,x) + table.at<Vec3i>(y,x-1) - table.at<Vec3i>(y-1,x-1);
            }
        }

        for (int y = r+1; y < h-r-1; y++){
            for (int x = r+1; x < w-r-1; x++){
                faceROI.at<Vec3i>(y,x) = table.at<Vec3i>(y+r, x+r) - table.at<Vec3i>(y+r, x-r-1) 
                                    - table.at<Vec3i>(y-r-1, x+r) + table.at<Vec3i>(y-r-1, x-r-1);
                faceROI.at<Vec3i>(y,x) /= area;
            }
        }

        img.convertTo(img, CV_8UC3);

        imshow("Blurred", img);
    }

    if (waitKey(0) == 'q'){
        return 0;
    }
    
    return 0;
}