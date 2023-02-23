/*
 * Authors:
 * 1. Jyothi Vishnu Vardhan Kolla.
 * 2. Vidya ganesh.
 * CS-5330 Spring 2023 semester.
 */

#include <iostream>
#include "distance_metrics.h"
#include "filters.h"
#include "csv_util.h"
#include <opencv2/opencv.hpp>

using namespace std;

// comparator function to sort vector pair based on value.
bool cmp(pair<string, double> &a, pair<string, double> &b) {
  return a.second < b.second;
}

// function to add label to the Image and display it.
int create_classified_image(cv::Mat &src,
							vector<pair<string, double>> &distances,
							const string &classification_method) {
  string label;
  if (classification_method=="Euclidean_dist") {
	label = distances[0].first;
  }

  cv::Point text_position(80, 80);
  int font_size = 1;
  cv::Scalar font_color(0, 0, 0);
  int font_weight = 2;
  cv::putText(src, label, text_position, cv::FONT_HERSHEY_COMPLEX, font_size, font_color, font_weight);
  return 0;
}
/*
 * A function that calculates scaled Euclidean distance for the test-image with all
   the images in the database and returns the label of the image with least distance.
 * Args1-testImg      : Path of the test Image.
 * Args-2-traindbpath : Path of the train database

 returns the label of the testImage as a string.
 */
int scaledEuclidean(cv::Mat &colorImg, cv::Mat &testImg, char traindbPath[]) {
  string result;
  vector<char *> filenames; // Vector to store filenames.
  vector<vector<double>> featureVectors; // Vector to store feature vectors.

  // get the feature vectors and associated labels.
  read_image_data_csv(traindbPath, filenames, featureVectors);
  cv::Mat target_features = get_moments(testImg);

  // find the euclidean distance and store them in a vector<pairs>
  vector<pair<string, double>> distances;
  for (int i = 0; i < featureVectors.size(); i++) {
	double euclidean_dist = 0;
	for (int j = 0; j < featureVectors[i].size(); j++) {
	  double x1 = 0.0;
	  double x2 = 0.0;
	  if (!::isnan(featureVectors[i][j]))
		x1 = featureVectors[i][j];
	  if (!::isnan(target_features.at<double>(j)))
		x2 = target_features.at<double>(j);

	  euclidean_dist += (x1 - x2)*(x1 - x2);
	}
	distances.emplace_back(filenames[i], euclidean_dist);
	sort(distances.begin(), distances.end(), cmp);
  }

  create_classified_image(colorImg, distances, "Euclidean_dist");
  return 0;
}