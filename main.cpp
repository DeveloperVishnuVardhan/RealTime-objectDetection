/*
 * Authors:
 * 1. Jyothi Vishnu Vardhan Kolla.
 * 2. Vidya ganesh.
 * CS-5330 Spring 2023 semester.
 */
#include <iostream>
#include<opencv2/opencv.hpp>
#include "filters.h"
#include <ctime>

using namespace std;
int main(int argc, char *argv[]) {
  // capture the video.
  cv::VideoCapture cap(0);
  if (!cap.isOpened()) {
	cout << "cannot open the camera" << endl;
	cin.get();
	return -1;
  }

  cv::Mat blurred_color_image, HSV_Image; // Mat object to store HSV_image.
  cv::Mat HSVthresholded_image; // Mat object to store thresholded image.
  cv::Mat Eroded_Hsv;
  cv::Mat thresholded_Image;

  while (true) {
	cv::Mat color_image; // Mat object to store original frame.
	bool bSucces = cap.read(color_image);

	// Break the while loop if frame cannot be captured.
	if (!bSucces) {
	  cout << "video camera is disconnected" << endl;
	  cin.get();
	  break;
	}
	cv::medianBlur(color_image, blurred_color_image, 5);
	cv::cvtColor(blurred_color_image, HSV_Image, cv::COLOR_BGR2HSV);
	threshold(HSV_Image, HSVthresholded_image);
	vector<vector<int>> Erosion_distance = GrassfireTransform(HSVthresholded_image);
	/*for (int i = 0; i < Erosion_distance.size(); i++) {
	  for (int j = 0; j < Erosion_distance[i].size();j++) {
		cout << Erosion_distance[i][j] << " ";
	  }
	  cout << endl;
	}*/
	Erosion(Erosion_distance, HSVthresholded_image, 2);
	vector<vector<int>> Dialation_distance = GrassfireTransform1(HSVthresholded_image);
	Dialation(Dialation_distance, HSVthresholded_image, 2);
	cv::cvtColor(HSVthresholded_image, thresholded_Image, cv::COLOR_HSV2BGR);

	// display the windows
	cv::namedWindow("color-Image", 1);
	cv::imshow("color-Image", color_image);
	cv::imshow("Threshold-Image", thresholded_Image);
	int k = cv::waitKey(10);

	if (k=='q') {
	  cv::destroyAllWindows();
	  break;
	} else if (k=='s') {
	  time_t now = time(0);
	  // convert now to string form
	  char *dt = ctime(&now);
	  string color_image_path =
		  "/Users/jyothivishnuvardhankolla/Desktop/Project-3Real-time-object-2DRecognition/Proj03Examples/Testing/color_erosion.png";
	  string thresholded_image_path =
		  "/Users/jyothivishnuvardhankolla/Desktop/Project-3Real-time-object-2DRecognition/Proj03Examples/Testing/threshold_after_erosion.png";
	  // save_color_image
	  cv::imwrite(color_image_path, color_image);
	  cv::imwrite(thresholded_image_path, thresholded_Image);
	}
	//break;
  }
}
