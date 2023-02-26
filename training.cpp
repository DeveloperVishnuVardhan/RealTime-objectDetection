/*
 * Authors:
 * 1. Jyothi Vishnu Vardhan Kolla.
 * 2. Vidya ganesh.
 * CS-5330 Spring 2023 semester.
 */

#include <iostream>
#include <experimental/filesystem>
#include<opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "filters.h"

using namespace std;
namespace fs = std::experimental::filesystem;

string get_label(string path, char delimeter) {
  vector<string> words;
  size_t first;
  size_t last = 0;

  while ((first = path.find_first_not_of(delimeter, last))
	  !=string::npos) {
	last = path.find(delimeter, first);
	words.push_back(path.substr(first, last - first));
  }
  return words[6];
}

void iterateDirectory(fs::path path) {
  for (const auto &entry : fs::directory_iterator(path)) {
	if (entry.is_directory()) {
	  // get the path and name of the subdirectory
	  fs::path subdir_path = entry.path();
	  // recursively call iterateDirectory for the subdirectory
	  iterateDirectory(subdir_path);
	} else {
	  if (entry.path().filename()!=".DS_Store")
		string img_path =
			entry.path();
	  cv::Mat color_image = cv::imread(entry.path()); // Mat object to store original frame.
	  if (color_image.empty()) {
		continue;
	  }
	  cv::Mat blurred_color_image, HSV_Image; // Mat object to store blurred, HSV_images.
	  cv::medianBlur(color_image, blurred_color_image, 5); // Blurring the color Image.
	  cv::cvtColor(blurred_color_image, HSV_Image, cv::COLOR_BGR2HSV); // Turing into HSV color space.
	  cv::Mat HSVthresholded_image; // Mat object to store thresholded image.
	  threshold(HSV_Image, HSVthresholded_image); // Threshold the Hsv image.
	  vector<vector<int>>
		  Erosion_distance = GrassfireTransform(HSVthresholded_image); // Vector to store Erosion distances.
	  Erosion(Erosion_distance, HSVthresholded_image, 5); // Perfrom Erosion.
	  vector<vector<int>>
		  Dialation_distance = GrassfireTransform1(HSVthresholded_image); // Vector to store Dialation distances.
	  Dialation(Dialation_distance, HSVthresholded_image, 5); // Perform Dialation.
	  cv::Mat thresholded_Image; // mat object to store final thresholded RGB Image.
	  cv::cvtColor(HSVthresholded_image, thresholded_Image, cv::COLOR_HSV2BGR);
	  //cout << entry.path();
	  string label = get_label(entry.path(), '/');
	  collect_data(thresholded_Image, label);
	}
  }
}

int main() {
  fs::path path = "/Users/jyothivishnuvardhankolla/Desktop/Project-3Real-time-object-2DRecognition/Data/Train";
  iterateDirectory(path);

  return 0;
}
