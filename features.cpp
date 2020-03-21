/**
 Commands To Run
-> g++ -std=c++11 features.cpp `pkg-config --libs --cflags opencv` -o features
-> ./features

**/

#include <opencv2/opencv.hpp>
#include <fstream>

using namespace std;
using namespace cv;


void fillHoles(Mat &mask)
{
    /*
     This hole filling algorithm is decribed in this post
     https://www.learnopencv.com/filling-holes-in-an-image-using-opencv-python-c/
     */

    Mat maskFloodfill = mask.clone();
    floodFill(maskFloodfill, cv::Point(0,0), Scalar(255));
    Mat mask2;
    bitwise_not(maskFloodfill, mask2);
    mask = (mask2 | mask);

}
void sharpness(Mat img,int w, int h)
{
  double variance;

  float ratio = 90000 /(img.cols * img.rows);
  Size size(300,300);
    Mat image;
    resize(img,image, size, ratio, ratio);
    Rect myROI(50,50,250,250);
    Mat image1=image(myROI);
    Mat gray ;
    cvtColor(image1,gray, COLOR_BGR2GRAY);
    Mat laplacianImage;
    variance =Laplacian(gray, laplacianImage, CV_64F).val[0];
    Scalar mean, stddev;
    meanStdDev(laplacianImage, mean, stddev, Mat());
    variance = stddev.val[0] * stddev.val[0];
    cout << "variance: " <<variance << endl;


}

int main(int argc, char** argv )
{
CascadeClassifier eyesCascade("haarcascade_eye.xml");
  ifstream ip("snap_scores.csv");
   if(!ip.is_open())std::cout<<"Could Not Open!"<< "\n";
    string snapPath,number,snapID,distance,badmatches,total,confidence,faceID,top,bot,left,right,userEstimate,extra;
    Mat snapImg;
    string snapDirectory="/home/wmk/IQA/snaps/";
    int size,t,l,w,h;
    double sharp;

   while(ip.good()){
     getline(ip,number,',');
     getline(ip,snapID,',');
     getline(ip,distance,',');
     getline(ip,badmatches,',');
     getline(ip,total,',');
     getline(ip,confidence,',');
     getline(ip,faceID,',');
     getline(ip,top,',');
     getline(ip,bot,',');
     getline(ip,left,',');
     getline(ip,right,',');
     getline(ip,userEstimate,'\n');
     if(snapID.find('"')<=snapID.size()){
     snapID.erase(snapID.find('"'),1);
}

     if((userEstimate== "0" )||(userEstimate == "1")||(userEstimate== "2" )||(userEstimate <= "3")||(userEstimate== "4" )||(userEstimate <= "5")){
       cout << "UserEstimate: " <<userEstimate << endl;
       snapPath=snapDirectory+snapID+".jpeg";
       snapImg=imread(snapPath);
       Mat img = imread(snapPath,CV_LOAD_IMAGE_COLOR);
       t=max(stoi(top), 0);
       l=max(stoi(left), 0);
       w=min(stoi(right), img.size().width )-l;
       h=min(stoi(bot), img.rows)-t;
       size  = w*h;
      cout << "width: " <<snapPath << endl;
      cout << "right: " <<snapID.find('"') << endl;
      cout << "image max width: " <<img.size().width  << endl;
      cout << "left: " <<l << endl;
      Rect crop=Rect(l,t,w,h);
      Mat croppedImage=img(crop);
      imshow("image",croppedImage);
      cvWaitKey(0);
      sharpness(croppedImage,w,h);

        // close the window


     }

   }

   ip.close();


    // Read image
    Mat img = imread("red_eyes2.jpg",CV_LOAD_IMAGE_COLOR);

    // Output image
    Mat imgOut = img.clone();

    // Load HAAR cascade

    // Detect eyes
    std::vector<Rect> eyes;
    eyesCascade.detectMultiScale( img, eyes, 1.3, 4, 0 |CASCADE_SCALE_IMAGE, Size(100, 100) );


    // For every detected eye
    for( size_t i = 0; i < eyes.size(); i++ )
    {

        // Extract eye from the image.
        Mat eye = img(eyes[i]);

        // Split eye image into 3 channels.
        vector<Mat>bgr(3);
        split(eye,bgr);

        // Simple red eye detector
        Mat mask = (bgr[2] > 150) & (bgr[2] > ( bgr[1] + bgr[0] ));

        // Clean mask -- 1) File holes 2) Dilate (expand) mask
        fillHoles(mask);
        dilate(mask, mask, Mat(), Point(-1, -1), 3, 1, 1);



        // Calculate the mean channel by averaging
        // the green and blue channels

        Mat mean = (bgr[0]+bgr[1])/2;
        mean.copyTo(bgr[2], mask);
        mean.copyTo(bgr[0], mask);
        mean.copyTo(bgr[1], mask);

        // Merge channels
        Mat eyeOut;
        cv::merge(bgr,eyeOut);

        // Copy the fixed eye to the output image.
        eyeOut.copyTo(imgOut(eyes[i]));

    }

    // Display Result
    imshow("Red Eyes", img);
    imshow("Red Eyes Removed", imgOut);
    waitKey(0);

}
