//
// Created by Jyothi vishnu vardhan Kolla on 2/22/23.
//

#ifndef MAIN_CPP__DISTANCE_METRICS_H_
#define MAIN_CPP__DISTANCE_METRICS_H_

#include <opencv2/opencv.hpp>
#include <experimental/filesystem>
using namespace std;
namespace fs = std::experimental::filesystem;
/*
 * A function that calculates scaled Euclidean distance for the test-image with all
   the images in the database and returns the label of the image with least distance.
 * Args1-testImg      : Path of the test Image.
 * Args-2-traindbpath : Path of the train database

 returns the label of the testImage as a string.
 */
vector<pair<string, double>> scaledEuclidean(cv::Mat &testImg, char traindbPath[], char threshtype[]);

// function to add label to the Image and display it.
int create_classified_image(cv::Mat &src,
							vector<pair<string, double>> &distances);
/*
 * A function that calculates scaled Euclidean distance for the test-image with all
   the images in the database and returns the label of the image with least distance.
 * Args-1-colorImg     : test Image RGB.
 * Args-2-testImg      : test Image thresholded.
 * Args-3-traindbpath  : Path of the train database

 returns a vector pair with label,count pairs in sorted order(descending).
 */
vector<pair<string, double>> knnClassifier(cv::Mat &testImg,
										   char traindbpath[],
										   int k_value,
										   string distance_metric,
										   char threshtype[]);

int evaluation(fs::path path, string distance_metric, char threshtype[]);
#endif //MAIN_CPP__DISTANCE_METRICS_H_
