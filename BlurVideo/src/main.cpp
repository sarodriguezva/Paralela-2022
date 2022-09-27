#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <iostream>

using namespace cv;
using namespace std;

Mat summed_table(Mat img, int w, int h){
    Mat table(w,h, CV_8UC3, Scalar(0,0,0));
    table.at<Vec3b>(0,0) = img.at<Vec3b>(0,0);

    for (int x = 1; x < w; x++){
        table.at<Vec3b>(x,0) += img.at<Vec3b>(x-1,0);
    }

    for (int y = 1; y < h; y++){
        table.at<Vec3b>(0,y) += img.at<Vec3b>(0,y-1);
    }

    for (int y = 1; y < h-1; y++){
        for (int x = 1; x < w-1; x++){
            table.at<Vec3b>(x,y) = img.at<Vec3b>(x,y) + table.at<Vec3b>(x,y-1) + table.at<Vec3b>(x-1,y) - table.at<Vec3b>(x-1,y-1);
        }
    }

    return table;
}

void myBlur(Mat face, Point tl, Point br, int r){
    int w = br.x - tl.x;
    int h = br.y - tl.y;

    int area = pow(2*r+1, 2);
    Mat table = summed_table(face, w, h);

    for (int y = r+1; y < h-r-1; y++){
        for (int x = r+1; x < w-r-1; x++){
            face.at<Vec3b>(x,y) = table.at<Vec3b>(x+r, y+r) - table.at<Vec3b>(x-r-1, y+r) 
                                - table.at<Vec3b>(x+r, y-r-1) + table.at<Vec3b>(x-r-1, y-r-1);
            face.at<Vec3b>(x,y) /= area;
        }
    }
}


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
            Point top_left = faces[i].tl();
            Point bot_right = faces[i].br();
            rectangle(img, top_left, bot_right, Scalar(0,0,0), 3);
            Mat faceROI = img(faces[i]);
            myBlur(faceROI, top_left, bot_right, 10);
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