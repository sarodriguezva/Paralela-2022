#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <iostream>

using namespace cv;
using namespace std;

//Blur Radio - CONSTANT
int r = 10;

/*
Auxiliar function. It creates a preffix sum array from any ROI.
It runs in O(w*h).

Creates an empty Opencv matrix (Mat) with signed-integers 3-tuples (CV_32SC3 pixel) in it.
The matrix is filled with the accumulative sum for each ROI's pixel (y,x) from (0,0) to (y-1,x-1).
Then, the matrix is returned.

Notes: 
+ CV_32SC3 is an OpenCV type and stands for 32bits Signed Channel-3,
which means that we are going with 3-Channel or RGB pixels whose respective value could vary from -2**31-1 to 2**31-1
+ This will be sufficient up to 1280x720p images. 
+ OpenCV doesn't define any larger type.
+ Vec3i is another way to refer to this 3-integer-tuples.

*/
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

/*
Main box-blur function.
It creates the preffix sum table and uses it for replacing in the original image the new averaged pixels.

Placing a new averaged pixel takes O(1) time.
Filling all the ROI in the original image takes O(w*h) time.
The function is adapted for supporting a variable blur radio.

Note:
+ Despite of the function receives a Region Of Interest (ROI), any change in this region yields
the change on the original image. So the blur filter applies in-place.

*/

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

/*
setROI function.
Box-blur has the problem of keeping unchanged the pixels who are at a distance less or equal than the radius from the ROI border.
In order to blurring all the detected face area, the function does some geometrical fixes and manages the corner cases.

The function runs in O(1) time and O(1) memory.
*/
Rect setROI(Rect faceROI){
    Rect ROI = faceROI;
    
    if(ROI.x - r >= 0 && ROI.y - r >= 0 && ROI.x + ROI.width + 2*r <= 640 && ROI.y + ROI.height + 2*r <= 480){
        ROI.x -= r;
        ROI.y -= r;
        ROI.width += 2*r;
        ROI.height += 2*r;
    }
    
    return ROI;
}

/*
Main function.
Loads the video and haar cascade face recognition data. Then it sets the output video container.
It process every input-video frame, resizes it to a 640x480 format and does a face recognition in grayscale mode.
Then, every face area is processed, blurred and then saved into the output video.
When the video is successfully saved, the funtion frees the memory.

Note:
+ Every image and ROI is originally in a CV_8UC3 format (8 unsigned bits and 3 channels).
+ The images need to be converted to CV_32SC3 for pixels arithmetic; then the results need to be returned to the original format.
+ The video can be played in runtime. The framerate depends on the computer.
*/
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
    
    VideoWriter video("../resources/videoOut.avi",cv::VideoWriter::fourcc('M','J','P','G'),10, Size(640,480));
    
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

            //rectangle(img, faces[i].tl(), faces[i].br(), Scalar(0, 255,0), 3);
            
            Point top_left = R.tl();
            Point bot_right = R.br();

            int w = bot_right.x - top_left.x;
            int h = bot_right.y - top_left.y;

            //rectangle(img, top_left, bot_right, Scalar(0,0,0), 3);

            Mat faceROI = img(R);
            faceROI.convertTo(faceROI, CV_32SC3);

            myBlur(faceROI, w, h);
            img.convertTo(img, CV_8UC3);
        }

        video.write(img);
        imshow("Blurred Face Detection", img);

        if (waitKey(1) == 'q'){
            break;
        }
    }

    cout << "Video guardado con éxito" << endl;
    cap.release();
    video.release();

    destroyAllWindows();
    
    return 0;
}