#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int r = 10;

Mat summed_table(Mat ROI, int w, int h){
    Mat table;
    table = Mat::zeros(ROI.size(), CV_32SC3);

    table.at<Vec3i>(0,0) = ROI.at<Vec3i>(0,0);
    for (int x = 1; x < w; x++){
        table.at<Vec3i>(0,x) = ROI.at<Vec3i>(0,x) + table.at<Vec3i>(0,x-1);
    }

    for (int y = 1; y < h; y++){
        table.at<Vec3i>(y,0) = ROI.at<Vec3i>(y,0) + table.at<Vec3i>(y-1,0);
    }

    for (int y = 1; y < h-1; y++){
        for (int x = 1; x < w-1; x++){
            table.at<Vec3i>(y,x) = ROI.at<Vec3i>(y,x) + table.at<Vec3i>(y-1,x) + table.at<Vec3i>(y,x-1) - table.at<Vec3i>(y-1,x-1);
        }
    }

    return table;
}

void myBlur(Mat face, int w, int h){
    int area = pow(2*r+1, 2);
    
    Mat table = summed_table(face, w, h);

    for (int y = r+1; y < h-r-1; y++){
        for (int x = r+1; x < w-r-1; x++){
            face.at<Vec3i>(y,x) = table.at<Vec3i>(y+r, x+r) - table.at<Vec3i>(y+r, x-r-1) 
                                - table.at<Vec3i>(y-r-1, x+r) + table.at<Vec3i>(y-r-1, x-r-1);
            face.at<Vec3i>(y,x) /= area;
        }
    }
}

Rect setROI(Rect faceROI){
    Rect ROI = faceROI;
    ROI.x -= r;
    ROI.y -= r;
    ROI.width += 2*r;
    ROI.height += 2*r;
    return ROI;
}


int main(int argc, char **argv){

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
            img.convertTo(img, CV_32SC3);

            Rect R = setROI(faces[i]);
            Point top_left = R.tl();
            Point bot_right = R.br();

            int w = bot_right.x - top_left.x;
            int h = bot_right.y - top_left.y;

            Mat faceROI = img(R);
            faceROI.convertTo(faceROI, CV_32SC3);

            myBlur(faceROI, w, h);
            img.convertTo(img, CV_8UC3);
        }

        imshow("Blurred Face Detection", img);

        if (waitKey(1) == 'q'){
            break;
        }
    }

    cap.release();

    destroyAllWindows();
    
    return 0;
}