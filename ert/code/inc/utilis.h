#ifndef __UTILIS_H__
#define __UTILIS_H__

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace ert{

typedef struct _split_node{
	int x;
	int y;
	int threshold;
}split_node;




typedef struct _train_sample{
public:
	cv::Rect rect;
	cv::Mat_<float> target_pts;
	cv::Mat_<float> cur_pts;
	int index;
	cv::Mat_<int> feature_values;
}train_sample;


void swap(train_sample *lhs, train_sample *rhs);

void copy(train_sample &lhs, train_sample &rhs);

cv::Mat_<float> normalization(train_sample &sample);

void unnormalization(train_sample &sample, cv::Mat_<float> &scale_rotate, cv::Mat_<float> &transform);

void get_similarity_transform(const cv::Mat_<float> &point_from, cv::Mat_<float> &point_to, cv::Mat_<float> &scale_rotate, cv::Mat_<float> &transform);

float mat_dot(cv::Mat_<float> mat1, cv::Mat_<float> mat2);
}


#endif