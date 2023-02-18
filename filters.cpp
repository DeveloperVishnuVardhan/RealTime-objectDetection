/*
 * Authors:
 * 1. Jyothi Vishnu Vardhan Kolla.
 * 2. Vidya ganesh.
 * CS-5330 Spring 2023 semester.
 */

#include<iostream>
#include<opencv2/opencv.hpp>
#include "filters.h"

using namespace std;

// function to fill pixels value in each hue, sat, val channels of hsv color space.
void fill_pixels(cv::Vec3b *rptr, int col, int h_value, int s_value, int v_value) {
  rptr[col][0] = h_value;
  rptr[col][1] = s_value;
  rptr[col][2] = v_value;
}
/*
 * Function to implement Thresholding.
 * src: Source Image on which the median filter needs to be applied.
 * dst: Destination container to store the Image after Applying median Filter.
 */
int threshold(cv::Mat &src, cv::Mat &dst) {
  dst = cv::Mat::zeros(src.rows, src.cols, CV_8UC3);
  // Define the lower and upper boundaries for white color in HSV space.
  int hue_low, sat_low, val_low, hue_high, sat_high, val_high;
  hue_low = 0;
  sat_low = 0;
  val_low = 0;

  hue_high = 179;
  sat_high = 25;
  val_high = 255;

  // Iterate through rows.
  for (int i = 0; i < src.rows; i++) {
	// create row pointers to access indexes.
	cv::Vec3b *rptr = src.ptr<cv::Vec3b>(i);
	cv::Vec3b *dptr = dst.ptr<cv::Vec3b>(i);
	// Iterate through columns.
	for (int j = 0; j < src.cols; j++) {
	  int hue = rptr[j][0];
	  int sat = rptr[j][1];
	  int val = rptr[j][2];
	  if ((hue >= hue_low && hue <= hue_high) && (sat >= sat_low && sat <= sat_high)
		  && (val >= val_low && val <= val_high)) {
		//cout << "thre" << endl;
		fill_pixels(dptr, j, 0, 0, 0);
	  } else {
		if (sat < 150) {
		  fill_pixels(dptr, j, 179, 25, 255);
		} else {
		  fill_pixels(dptr, j, 179, 25, 235);
		}
	  }
	}
  }
  return 0;
}

/*
 * Function to implement 8-connected Grassfire Transform.
 * src: Thresholded HSV Image,
 */
vector<vector<int>> GrassfireTransform1(cv::Mat &src) {
  vector<vector<int>> dist(src.rows, vector<int>(src.cols, 0));
  // pass-1
  for (int i = 0; i < src.rows; i++) {
	// create row pointers to access indexes.
	cv::Vec3b *rptr = src.ptr<cv::Vec3b>(i);
	for (int j = 0; j < src.cols; j++) {
	  int hue = rptr[j][0];
	  int sat = rptr[j][1];
	  int val = rptr[j][2];

	  if (hue!=0 && sat!=0 && val!=0) {
		dist[i][j] = 0;
	  } else {
		// Becareful to handle out of bound cases.
		int top, left, top_dist, top_left_dist, top_right_dist, left_dist;
		top = i - 1;
		left = j - 1;

		if (top < 0) {
		  top_dist = 0;
		  top_left_dist = 0;
		  top_right_dist = 0;
		} else if (left < 0) {
		  left_dist = 0;
		  top_left_dist = 0;
		} else {
		  top_dist = dist[i - 1][j];
		  left_dist = dist[i][j - 1];
		  top_left_dist = dist[i - 1][j - 1];
		  top_right_dist = dist[i - 1][j + 1];
		}
		int min_1 = min(top_dist, left_dist);
		int min_2 = min(top_left_dist, top_right_dist);
		dist[i][j] = min(min_1, min_2) + 1; // min of neigh pixel + 1
	  }
	}
  }

  // pass-2: Iterate in reverse direction
  for (int i = dist.size() - 1; i >= 0; --i) {
	for (int j = dist[i].size() - 1; j >= 0; --j) {
	  int down, right, down_dist, right_dist, down_right;
	  down = i + 1;
	  right = j + 1;

	  if (down >= dist.size()) {
		down_dist = 0;
		down_right = 0;
	  } else if (right >= dist[i].size()) {
		right_dist = 0;
		down_right = 0;
	  } else {
		down_dist = dist[i + 1][j];
		right_dist = dist[i][j + 1];
		down_right = dist[i + 1][j + 1];
	  }
	  int curr_dis = dist[i][j];
	  int nei_dis = min(down_dist, min(right_dist, down_right)) + 1;
	  dist[i][j] = min(curr_dis, nei_dis);
	}
  }

  return dist;
}

/*
 * Function to perform 4-connected Grassfire Transform.
 */
vector<vector<int>> GrassfireTransform(cv::Mat &src) {
  vector<vector<int>> dist(src.rows, vector<int>(src.cols, 0));
  // pass-1
  for (int i = 0; i < src.rows; i++) {
	// create row pointers to access indexes.
	cv::Vec3b *rptr = src.ptr<cv::Vec3b>(i);
	for (int j = 0; j < src.cols; j++) {
	  int hue = rptr[j][0];
	  int sat = rptr[j][1];
	  int val = rptr[j][2];

	  if (hue==0 && sat==0 && val==0) {
		dist[i][j] = 0;
	  } else {
		// Becareful to handle out of bound cases.
		int top, left, top_dist, left_dist;
		top = i - 1;
		left = j - 1;

		if (top < 0) top_dist = 0;
		else if (left < 0) left_dist = 0;
		else {
		  top_dist = dist[i - 1][j];
		  left_dist = dist[i][j - 1];
		}
		dist[i][j] = min(top_dist, left_dist) + 1; // min of neigh pixel + 1
	  }
	}
  }

  // pass-2: Iterate in reverse direction
  for (int i = dist.size() - 1; i >= 0; --i) {
	for (int j = dist[i].size() - 1; j >= 0; --j) {
	  int down, right, down_dist, right_dist;
	  down = i + 1;
	  right = j + 1;

	  if (down >= dist.size()) down_dist = 0;
	  else if (right >= dist[i].size()) right_dist = 0;
	  else {
		down_dist = dist[i + 1][j];
		right_dist = dist[i][j + 1];
	  }
	  int curr_dis = dist[i][j];
	  int nei_dis = min(down_dist, right_dist) + 1;
	  dist[i][j] = min(curr_dis, nei_dis);
	}
  }

  return dist;
}

/*
 * Funtion to perform Erosion.
 *
 */
int Erosion(vector<vector<int>> &distances, cv::Mat &src, int erosion_length) {
  //dst = cv::Mat::zeros(src.rows, src.size, CV_8UC3);
  for (int i = 0; i < src.rows; i++) {
	// create a row pointer to access pixesls
	cv::Vec3b *rptr = src.ptr<cv::Vec3b>(i);
	//cv::Vec3b *dptr = dst.ptr<cv::Vec3b>(i);
	for (int j = 0; j < src.cols; j++) {
	  if (distances[i][j] <= erosion_length) {
		rptr[j][0] = 0;
		rptr[j][1] = 0;
		rptr[j][2] = 0;
	  }
	}
  }
  return 0;
}

/*
 * Funtion to perform Dialation.
 *
 */
int Dialation(vector<vector<int>> &distances, cv::Mat &src, int erosion_length) {
  //dst = cv::Mat::zeros(src.rows, src.size, CV_8UC3);
  for (int i = 0; i < src.rows; i++) {
	// create a row pointer to access pixesls
	cv::Vec3b *rptr = src.ptr<cv::Vec3b>(i);
	//cv::Vec3b *dptr = dst.ptr<cv::Vec3b>(i);
	for (int j = 0; j < src.cols; j++) {
	  if (distances[i][j] <= erosion_length) {
		rptr[j][0] = 179;
		rptr[j][1] = 25;
		rptr[j][2] = 255;
	  }
	}
  }
  return 0;
}