/*
 * Authors:
 * 1. Jyothi Vishnu Vardhan Kolla.
 * 2. Vidya ganesh.
 * CS-5330 Spring 2023 semester.
 */
#include <iostream>
#include<opencv2/opencv.hpp>
#include "distance_metrics.h"

using namespace std;
int main(int argc, char *argv[]) {
  // capture the video.
  cv::VideoCapture cap(0);
  if (!cap.isOpened()) {
	cout << "cannot open the camera" << endl;
	cin.get();
	return -1;
  }

  while (true) {
	cv::Mat test_color_img; // Mat object to store original frame.
	bool bSucces = cap.read(test_color_img);

	// Break the while loop if frame cannot be captured.
	if (!bSucces) {
	  cout << "video camera is disconnected" << endl;
	  cin.get();
	  break;
	}
	char train_db[256]; // storing path to train database.
	char classifier[256]; // storing classifier type.
	::strcpy(train_db,
			 "/Users/jyothivishnuvardhankolla/Desktop/Project-3Real-time-object-2DRecognition/Project-3/train.csv");
	::strcpy(classifier, argv[1]);

	vector<pair<string, double>> distances; // Vector to store distances from each Image in database to test image.

	if (::strcmp(classifier, "scaledeuclidean")==0)
	  distances = scaledEuclidean(test_color_img, test_color_img, train_db);
	if (::strcmp(classifier, "knn")==0) {
	  distances = knnClassifier(test_color_img, test_color_img, train_db, 15);
	}

	for (int i = 0; i < distances.size(); i++) {
	  cout << distances[i].first << distances[i].second << endl;
	}

	create_classified_image(test_color_img, distances);
	// display the windows
	cv::imshow("classified-image", test_color_img);
	int k = cv::waitKey(10);

	if (k=='q') {
	  cv::destroyAllWindows();
	  break;
	}
  }
}
