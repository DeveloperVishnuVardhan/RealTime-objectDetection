/*
 * Authors:
 * 1. Jyothi Vishnu Vardhan Kolla.
 * 2. Vidya ganesh.
 * CS-5330 Spring 2023 semester.
 */
#include <iostream>
#include<opencv2/opencv.hpp>
#include "filters.h"

using namespace std;
int main(int argc, char *argv[]) {
  string img_path =
	  "/Users/jyothivishnuvardhankolla/Desktop/Project-3Real-time-object-2DRecognition/Proj03Examples/IMG_1013.png";
  cv::Mat color_image = cv::imread(img_path); // Mat object to store original frame.
  if (color_image.empty()) {
	cout << "could not load and display the image" << endl;
	cin.get(); // wait for a key stroke
	exit(-1);
  }
  cv::Mat blurred_color_image, HSV_Image; // Mat object to store HSV_image.
  cv::medianBlur(color_image, blurred_color_image, 5);
  cv::cvtColor(blurred_color_image, HSV_Image, cv::COLOR_BGR2HSV);
  cv::Mat HSVthresholded_image; // Mat object to store thresholded image.
  threshold(HSV_Image, HSVthresholded_image);
  cv::Mat thresholded_Image;
  cv::cvtColor(HSVthresholded_image, thresholded_Image, cv::COLOR_HSV2BGR);

  while (true) {
	cv::namedWindow("color-Image", 1);
	cv::imshow("color-Image", color_image);
	cv::imshow("Threshold-Image", thresholded_Image);
	int k = cv::waitKey(0);

	if (k=='q') {
	  cv::destroyAllWindows();
	  break;
	}

  }
}
