/*Loading a cascade classifier and  finding objects (Face + eyes) in a video stream - Using LBPH (Local Binary Pattern Histogram) Method*/
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

void detectAndDisplay( Mat frame );

String face_cascade_name = "lbpcascade_frontalface.xml";
String eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
string window_name = "DTU Face Detection Module - Using LBP";

int main( void )
{
  VideoCapture capture;
  Mat frame;

  // 1. Loading the cascade classifer
  if( !face_cascade.load( face_cascade_name ) ){ printf("(!)Error loading\n"); return -1; };
  if( !eyes_cascade.load( eyes_cascade_name ) ){ printf("(!)Error loading\n"); return -1; };

  //2. Reading the video stream
  capture.open( -1 );
  if( capture.isOpened() )
  {
    for(;;)
    {
      capture >> frame;

      //3. Applying the classifier to the frame
      if( !frame.empty() )
       { detectAndDisplay( frame ); }
      else
       { printf("No captured frame -- Break!"); break; }

      int c = waitKey(10);
      if( (char)c == 'c' ) { break; }

    }
  }
  return 0;
}

void detectAndDisplay( Mat frame )
{
   std::vector<Rect> faces;
   Mat frame_gray;

   // Converts frame from BGR Format to GrayScale
   cvtColor( frame, frame_gray, COLOR_BGR2GRAY );
   
   // Contrast adjustment using image histogram
   equalizeHist( frame_gray, frame_gray );

   // Detect faces
   face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0, Size(80, 80) );

   for( size_t i = 0; i < faces.size(); i++ )
    {
      Mat faceROI = frame_gray( faces[i] );
      std::vector<Rect> eyes;

      //In each face, detect eyes
      eyes_cascade.detectMultiScale( faceROI, eyes, 1.1, 2, 0 |CV_HAAR_SCALE_IMAGE, Size(30, 30) );
      if( eyes.size() == 2)
      {
         //Draw the faces
         Point center( faces[i].x + faces[i].width/2, faces[i].y + faces[i].height/2 );
         ellipse( frame, center, Size( faces[i].width/2, faces[i].height/2), 0, 0, 360, Scalar( 0, 0, 255 ), 2, 8, 0 );

         for( size_t j = 0; j < eyes.size(); j++ )
          { //Draw the eyes
            Point eye_center( faces[i].x + eyes[j].x + eyes[j].width/2, faces[i].y + eyes[j].y + eyes[j].height/2 );
            int radius = cvRound( (eyes[j].width + eyes[j].height)*0.25 );
            circle( frame, eye_center, radius, Scalar( 255, 0, 0 ), 3, 8, 0 );
          }
       }

    }
   //Display the result
   imshow( window_name, frame );
}
