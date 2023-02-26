/*
 * Authors:
 * 1. Jyothi Vishnu Vardhan Kolla.
 * 2. Vidya ganesh.
 * CS-5330 Spring 2023 semester.
 */

#include<iostream>
#include<opencv2/opencv.hpp>
#include "filters.h"
#include "csv_util.h"
#include <cmath>

using namespace std;

// comparator function to sort vector pair based on value.
bool cmp(pair<int, int> &a, pair<int, int> &b) {
  return a.second < b.second;
}

// function to fill pixels value in each hue, sat, val channels of hsv color space.
void fill_pixels(cv::Vec3b *rptr, int col, int h_value, int s_value, int v_value) {
  rptr[col][0] = h_value;
  rptr[col][1] = s_value;
  rptr[col][2] = v_value;
}

// helper function to get moment values given a thresholded RGB image.
vector<double> get_moments(cv::Mat &src) {
  cv::Mat Grayscale_img;
  cv::cvtColor(src, Grayscale_img, cv::COLOR_BGR2GRAY);
  vector<double> features;
  cv::Mat thresholded_grayscale_img;
  cv::threshold(Grayscale_img, thresholded_grayscale_img, 128, 255, cv::THRESH_BINARY);
  cv::Moments moments = cv::moments(thresholded_grayscale_img, false);
  double huMoments[7];
  cv::HuMoments(moments, huMoments);
  for (int i = 0; i < 7; i++) {
	features.push_back(-1*copysign(1.0, huMoments[i])*log10(abs(huMoments[i])));
  }
  return features;
}

/*
 * Function to implement Thresholding.
 * src: Source HSV Image on which the Thresholding need to be applied.
 * dst: Destination container to store the Image after Applying Thresholding.
 */
int threshold(cv::Mat &src, cv::Mat &dst) {
  dst = cv::Mat::zeros(src.rows, src.cols, CV_8UC3);
  // Define the lower and upper boundaries for white color in HSV space.
  int hue_low, sat_low, val_low, hue_high, sat_high, val_high;
  hue_low = 0;
  sat_low = 0;
  val_low = 180;

  hue_high = 250;
  sat_high = 30;
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
 * Function to implement 8-connected Grassfire Transform Given a thresholded HSV Image.
 * src: Thresholded HSV Image.
 * Return a 2d vector where each cell contains distance to nearest foreground pixel.
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
 * Function to perform 4-connected Grassfire Transform given HSV thresholded Image.
 * src: Thresholded HSV Image.
 * Return a 2d vector where each cell contains distance to nearest background pixel.
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
 * Arg1-distances: distances matrix after performing Grassfire transform.
 * Arg-2 erosion_length: Number of erosions to perform.
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
 * Arg1-distances: distances matrix after performing Grassfire transform.
 * Arg-2 erosion_length: Number of dialtaions to perform.
 */
int Dialation(vector<vector<int>> &distances, cv::Mat &src, int dialation_length) {
  //dst = cv::Mat::zeros(src.rows, src.size, CV_8UC3);
  for (int i = 0; i < src.rows; i++) {
	// create a row pointer to access pixesls
	cv::Vec3b *rptr = src.ptr<cv::Vec3b>(i);
	//cv::Vec3b *dptr = dst.ptr<cv::Vec3b>(i);
	for (int j = 0; j < src.cols; j++) {
	  if (distances[i][j] <= dialation_length) {
		rptr[j][0] = 179;
		rptr[j][1] = 25;
		rptr[j][2] = 255;
	  }
	}
  }
  return 0;
}

/*
 * Function to perform Segmentation.
 * Arg-1-src: Thresholded RGB image on which segmentation to be performed.
 * returns a new Image where the Top-N components are filled with random colors.
 */
cv::Mat SegmentImage(cv::Mat &src) {
  // performing segmentation(connected-component analysis).
  cv::Mat Thresholded_Grayscale_img;
  cv::Mat segmented_image = src.clone();
  cv::cvtColor(src, Thresholded_Grayscale_img, cv::COLOR_BGR2GRAY);
  cv::Mat ImageIds, stats_matrix, centroids;
  int num_components =
	  cv::connectedComponentsWithStats(Thresholded_Grayscale_img, ImageIds, stats_matrix, centroids, 4);

  // Generate a random color for each component
  cv::RNG rng(0xFFFFFFFF);
  std::vector<cv::Scalar> colors(num_components);
  for (int i = 1; i < num_components; i++) {
	colors[i] = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
  }

  // Label each component with a different color
  for (int i = 1; i < num_components; i++) {
	cv::Rect rect(stats_matrix.at<int>(i, cv::CC_STAT_LEFT), stats_matrix.at<int>(i, cv::CC_STAT_TOP),
				  stats_matrix.at<int>(i, cv::CC_STAT_WIDTH), stats_matrix.at<int>(i, cv::CC_STAT_HEIGHT));
	cv::rectangle(segmented_image, rect, colors[i], 2);
  }
  return segmented_image;
}

/*
 * Function to compute the moments of regions in a given image.
 * Args-1-src  : Thrsholded RGB Image.
 * Returns a new Image with central Axis drawn on it.
 */
cv::Mat calculate_moments(cv::Mat &src) {
  cv::Mat Thresholded_Grayscale_img, central_moment_image;
  central_moment_image = src.clone();
  cv::cvtColor(src, Thresholded_Grayscale_img, cv::COLOR_BGR2GRAY);

  // calculate the moments of the Image.
  cv::Moments moments = cv::moments(Thresholded_Grayscale_img);

  // caluclate the Hu moments of the Image.
  cv::Mat hu_moments;
  cv::HuMoments(moments, hu_moments);

  // calculate the axis of central moments.
  double u20 = hu_moments.at<double>(2, 0);
  double u02 = hu_moments.at<double>(0, 2);
  double u11 = hu_moments.at<double>(1, 1);
  double theta = 0.5*::atan2(2*u11, u20 - u02);

  // calculate centroid of the image.
  double x_cor = moments.m10/moments.m00;
  double y_cor = moments.m01/moments.m00;

  // calculate the endpoints of the line for the axis of central moments.
  double cos_theta = cos(theta);
  double sin_theta = sin(theta);

  double x1_point = x_cor - 100*sin_theta;
  double y1_point = y_cor - 100*cos_theta;
  double x2_point = x_cor + 100*sin_theta;
  double y2_point = y_cor + 100*cos_theta;

  double angle = 0.5*(::atan((2*u11)/(u20 - u02)));


  // Draw axis of central moment on the Image.
  cv::line(central_moment_image,
		   cv::Point(x1_point, y1_point),
		   cv::Point(x2_point, y2_point),
		   cv::Scalar(0, 0, 255),
		   2);

  // draw a oriented bounding box.
  cv::Point centerPoint(x_cor, y_cor);
  cv::Scalar color = cv::Scalar(234, 234, 123);

  // find the size of rectangle.
  cv::Mat ImageIds, stats_matrix, centroids;
  int num_components =
	  cv::connectedComponentsWithStats(Thresholded_Grayscale_img, ImageIds, stats_matrix, centroids, 4);
  double height = stats_matrix.at<int>(2, cv::CC_STAT_HEIGHT);
  double width = stats_matrix.at<int>(2, cv::CC_STAT_HEIGHT);

  cout << height << " " << width << endl;

  // create a rotated rectangle.
  cv::RotatedRect rotatedRectangle(centerPoint, cv::Size2f(width, height), angle);

  // We take the edges that OpenCV calculated for us
  cv::Point2f vertices2f[4];
  rotatedRectangle.points(vertices2f);

  // Convert them so we can use them in a fillConvexPoly
  for (int i = 0; i < 4; ++i) {
	cv::line(central_moment_image, vertices2f[i], vertices2f[(i + 1)%4], cv::Scalar(0, 255, 0), 2);
  }

  // Now we can fill the rotated rectangle with our specified color
  cv::Rect brect = rotatedRectangle.boundingRect();
  return central_moment_image;
}

/*
 * Function that stores Moments as fearues in a csv file given a Thresholded RGB Image.
 * Args-1-src  : Thresholded RGB Image.
 */
int collect_data(cv::Mat &src, string label) {
  vector<double> features;
  features = get_moments(src);
  // Taking the label from console.
  char filename[256] =
	  "/Users/jyothivishnuvardhankolla/Desktop/Project-3Real-time-object-2DRecognition/Project-3/train.csv";
  if (label=="") {
	cin >> label;
  }
  char class_type[256];
  ::strcpy(class_type, label.c_str());
  append_image_data_csv(filename, class_type, features, 0);
  return 0;
}