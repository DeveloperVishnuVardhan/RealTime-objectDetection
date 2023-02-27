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
#include "csv_util.h"
#include <experimental/filesystem>
using namespace std;
namespace fs = std::experimental::filesystem;

int main(int argc, char *argv[]) {
  // Terminate if invalid number of command line arguements.
  if (argc!=6) {
	cout << "Invalid number of command line arguements" << endl;
	cin.get(); // wait for key press.
	exit(-1);
  }

  char train_db[256]; // storing path to train database.
  char target_image[256]; // storing path to test image.
  char classifier[256]; // storing classifier type.
  char evaluate[256]; // stores the status of evaluate.
  char distance_measure[256]; // store the distance measure to use.
  char thresh_type[256];
  ::strcpy(target_image, argv[1]);
  ::strcpy(train_db,
		   "/Users/jyothivishnuvardhankolla/Desktop/Project-3Real-time-object-2DRecognition/Project-3/train.csv");
  ::strcpy(classifier, argv[2]);
  ::strcpy(evaluate, argv[3]);
  ::strcpy(distance_measure, argv[4]);
  ::strcpy(thresh_type, argv[5]);

  cv::Mat test_color_img = cv::imread(target_image); // read the image.
  vector<pair<string, double>> distances; // Vector to store distances from each Image in database to test image.

  if (::strcmp(classifier, "scaledeuclidean")==0)
	distances = scaledEuclidean(test_color_img, train_db, thresh_type);
  else if (::strcmp(classifier, "knn")==0) {
	cout << "using knn";
	distances = knnClassifier(test_color_img, train_db, 15, distance_measure, thresh_type);
  }
  for (int i = 0; i < distances.size(); i++) {
	cout << distances[i].first << distances[i].second << endl;
  }

  // if needed to evaluate.
  if (::strcmp(evaluate, "yes")==0) {
	fs::path path = "/Users/jyothivishnuvardhankolla/Desktop/Project-3Real-time-object-2DRecognition/Data/Test";
	evaluation(path, distance_measure, thresh_type);
  }

  // creating classified Image.
  create_classified_image(test_color_img, distances);
  cv::imshow("classified-image", test_color_img);
  cv::waitKey(0);
}


