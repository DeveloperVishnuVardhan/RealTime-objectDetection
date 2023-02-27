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

// comparator function to sort vector pair based on value in ascending order.
bool cmp(pair<string, double> &a, pair<string, double> &b) {
  return a.second < b.second;
}

// comparator function to sort vector pair based on value in descending order.
bool cmp1(pair<string, double> &a, pair<string, double> &b) {
  return a.second > b.second;
}

// function to add label to the Image and display it.
int create_classified_image(cv::Mat &src,
							vector<pair<string, double>> &distances) {
  string label;
  label = distances[0].first;

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
 * Args-1-colorImg    : test Image RGB.
 * Args-3-testImg     : test Image thresholded.
 * Args-3-traindbpath : Path of the train database

 returns a Vector pair of distances with lable, distances as pairs in sorted order.
 */
vector<pair<string, double>> scaledEuclidean(cv::Mat &testImg, char traindbPath[]) {
  vector<char *> filenames; // Vector to store filenames.
  vector<vector<double>> featureVectors; // Vector to store feature vectors.

  // get the feature vectors and associated labels.
  read_image_data_csv(traindbPath, filenames, featureVectors);
  vector<double> target_features = get_moments(testImg);
  // find the euclidean distance and store them in a vector<pairs>
  vector<pair<string, double>> distances;
  for (int i = 0; i < featureVectors.size(); i++) {
	double euclidean_dist = 0;
	for (int j = 0; j < featureVectors[i].size(); j++) {
	  double x1 = featureVectors[i][j];
	  double x2 = target_features[j];
	  euclidean_dist += (x1 - x2)*(x1 - x2);
	}
	distances.emplace_back(filenames[i], sqrt(euclidean_dist));
  }
  // sort the distances.
  sort(distances.begin(), distances.end(), cmp);
  for (int i = 0; i < distances.size(); i++) {
	cout << distances[i].first << ":" << distances[i].second << endl;
  }
  return distances;
}

/*
 * A function that calculates scaled Euclidean distance for the test-image with all
   the images in the database and returns the label of the image with least distance.
 * Args-1-colorImg     : test Image RGB.
 * Args-2-testImg      : test Image thresholded.
 * Args-3-traindbpath  : Path of the train database

 returns a vector pair with label,count pairs in sorted order(descending).
 */
vector<pair<string, double>> knnClassifier(cv::Mat &testImg, char traindbpath[], int k_value) {
  vector<pair<string, double>> euclidean_distances; // Vector pair to store euclidean-distances.
  euclidean_distances = scaledEuclidean(testImg, traindbpath); // compute euclidean distances.

  unordered_map<string, double> counts; // Hashmap to store counts of first k-closest labels
  for (int i = 0; i < k_value; i++) {
	string key = euclidean_distances[i].first;
	if (counts.find(key)==counts.end())
	  counts[key] = 1;
	else
	  counts[key] += 1;
  }

  unordered_map<string, double>::iterator itr; // iterating through the hash_map.
  vector<pair<string, double>> sorted_counts; // storing sorted hash_map as vector of pairs.
  for (itr = counts.begin(); itr!=counts.end(); itr++)
	sorted_counts.emplace_back(itr->first, itr->second);
  sort(sorted_counts.begin(), sorted_counts.end(), cmp1);
  return sorted_counts;
}
