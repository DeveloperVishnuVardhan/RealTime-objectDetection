/*
 * Authors:
 * 1. Jyothi Vishnu Vardhan Kolla.
 * 2. Vidya ganesh.
 * CS-5330 Spring 2023 semester.
 */
#include <iostream>
#include<opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "filters.h"

using namespace std;
int main(int argc, char *argv[]) {
  string img_path =
	  "/Users/jyothivishnuvardhankolla/Desktop/Project-3Real-time-object-2DRecognition/Data/Train/Mouse/IMG_1319.png";
  cv::Mat color_image = cv::imread(img_path); // Mat object to store original frame.
  if (color_image.empty()) {
	cout << "could not load and display the image" << endl;
	cin.get(); // wait for a key stroke
	exit(-1);
  }
  cv::Mat blurred_color_image, HSV_Image; // Mat object to store blurred, HSV_images.
  cv::medianBlur(color_image, blurred_color_image, 5); // Blurring the color Image.
  cv::cvtColor(blurred_color_image, HSV_Image, cv::COLOR_BGR2HSV); // Turing into HSV color space.
  cv::Mat HSVthresholded_image; // Mat object to store thresholded image.
  threshold(HSV_Image, HSVthresholded_image); // Threshold the Hsv image.
  vector<vector<int>> Erosion_distance = GrassfireTransform(HSVthresholded_image); // Vector to store Erosion distances.
  Erosion(Erosion_distance, HSVthresholded_image, 5); // Perfrom Erosion.
  vector<vector<int>>
	  Dialation_distance = GrassfireTransform1(HSVthresholded_image); // Vector to store Dialation distances.
  Dialation(Dialation_distance, HSVthresholded_image, 5); // Perform Dialation.
  cv::Mat thresholded_Image; // mat object to store final thresholded RGB Image.
  cv::cvtColor(HSVthresholded_image, thresholded_Image, cv::COLOR_HSV2BGR);
  collect_data(thresholded_Image);
  cv::Mat Segmented_Image = SegmentImage(thresholded_Image); // perform segmentation.

  // Calculating axis of least central moment.
  cv::Mat central_moment_image = calculate_moments(thresholded_Image);
  collect_data(color_image);

  while (true) {
	cv::namedWindow("color-Image", 1);
	cv::imshow("color-Image", color_image);
	cv::imshow("Threshold-Image", thresholded_Image);
	cv::imshow("segmented-Image", Segmented_Image);
	cv::imshow("Central-moments", central_moment_image);
	//cv::imshow("Segmented-Image", color_image);
	int k = cv::waitKey(0);

	if (k=='q') { // destroy all windows when 'q' is pressed.
	  cv::destroyAllWindows();
	  break;
	} else if (k=='n') { // save the image features to db when 'N' is pressed.
	  collect_data(thresholded_Image);
	}

  }
}