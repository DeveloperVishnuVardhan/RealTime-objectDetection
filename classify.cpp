/*
 * Authors:
 * 1. Jyothi Vishnu Vardhan Kolla.
 * 2. Vidya ganesh.
 * CS-5330 Spring 2023 semester.
 * This source code takes a test-Image as a cmd line arg and displays the classified
 * label of the Image as a window
 */

#include <iostream>
#include <cstring>
#include "distance_metrics.h"
#include "filters.h"
using namespace std;

int main(int argc, char *argv[]) {
  // Terminate if invalid number of command line arguements.
  if (argc!=2) {
	cout << "Invalid number of command line arguements" << endl;
	cin.get(); // wait for key press.
	exit(-1);
  }

  char train_db[256];
  char target_image[256];
  ::strcpy(target_image, argv[1]);
  ::strcpy(train_db,
		   "/Users/jyothivishnuvardhankolla/Desktop/Project-3Real-time-object-2DRecognition/Project-3/train.csv");

  cv::Mat test_color_img = cv::imread(target_image); // read the image.
  cv::Mat blurred_color_image, HSV_Image; // Mat object to store HSV_image.
  cv::medianBlur(test_color_img, blurred_color_image, 5);
  cv::cvtColor(blurred_color_image, HSV_Image, cv::COLOR_BGR2HSV);
  cv::Mat HSVthresholded_image; // Mat object to store thresholded image.
  threshold(HSV_Image, HSVthresholded_image);
  vector<vector<int>> Erosion_distance = GrassfireTransform(HSVthresholded_image); // Vector to store Erosion distances.
  Erosion(Erosion_distance, HSVthresholded_image, 3); // Perfrom Erosion.
  vector<vector<int>>
	  Dialation_distance = GrassfireTransform1(HSVthresholded_image); // Vector to store Dialation distances.
  Dialation(Dialation_distance, HSVthresholded_image, 3); // Perform Dialation.
  cv::Mat thresholded_Image;
  cv::cvtColor(HSVthresholded_image, thresholded_Image, cv::COLOR_HSV2BGR);

  scaledEuclidean(test_color_img, thresholded_Image, train_db);
  cv::imshow("classified-image", test_color_img);
  cv::waitKey(0);
}


