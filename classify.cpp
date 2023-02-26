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
  if (argc!=3) {
	cout << "Invalid number of command line arguements" << endl;
	cin.get(); // wait for key press.
	exit(-1);
  }

  char train_db[256]; // storing path to train database.
  char target_image[256]; // storing path to test image.
  char classifier[256]; // storing classifier type.
  ::strcpy(target_image, argv[1]);
  ::strcpy(train_db,
		   "/Users/jyothivishnuvardhankolla/Desktop/Project-3Real-time-object-2DRecognition/Project-3/train.csv");
  ::strcpy(classifier, argv[2]);

  cv::Mat test_color_img = cv::imread(target_image); // read the image.

  vector<pair<string, double>> distances; // Vector to store distances from each Image in database to test image.

  if (::strcmp(classifier, "scaledeuclidean")==0)
	distances = scaledEuclidean(test_color_img, test_color_img, train_db);
  if (::strcmp(classifier, "knn")==0) {
	distances = knnClassifier(test_color_img, test_color_img, train_db, 5);
  }

  for (int i = 0; i < distances.size(); i++) {
	cout << distances[i].first << distances[i].second << endl;
  }

  create_classified_image(test_color_img, distances);
  cv::imshow("classified-image", test_color_img);
  cv::waitKey(0);
}


