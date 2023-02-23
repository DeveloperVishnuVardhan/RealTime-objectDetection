//
// Created by Jyothi vishnu vardhan Kolla on 2/22/23.
//

#ifndef MAIN_CPP__DISTANCE_METRICS_H_
#define MAIN_CPP__DISTANCE_METRICS_H_

#include <opencv2/opencv.hpp>
/*
 * A function that calculates scaled Euclidean distance for the test-image with all
   the images in the database and returns the label of the image with least distance.
 * Args1-testImg      : Path of the test Image.
 * Args-2-traindbpath : Path of the train database

 returns the label of the testImage as a string.
 */
int scaledEuclidean(cv::Mat &colorImg, cv::Mat &testImg, char traindbPath[]);
#endif //MAIN_CPP__DISTANCE_METRICS_H_
