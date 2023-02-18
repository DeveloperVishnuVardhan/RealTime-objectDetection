/*
 * Authors:
 * 1. Jyothi Vishnu Vardhan Kolla.
 * 2. Vidya ganesh.
 * CS-5330 Spring 2023 semester.
 */

#ifndef MAIN_CPP__FILTERS_H_
#define MAIN_CPP__FILTERS_H_

int threshold(cv::Mat &src, cv::Mat &dst);
std::vector<std::vector<int>> GrassfireTransform(cv::Mat &src);
int Erosion(std::vector<std::vector<int>> &distances, cv::Mat &src, int erosion_length);
std::vector<std::vector<int>> GrassfireTransform1(cv::Mat &src);
int Dialation(std::vector<std::vector<int>> &distances, cv::Mat &src, int erosion_length);
#endif //MAIN_CPP__FILTERS_H_
