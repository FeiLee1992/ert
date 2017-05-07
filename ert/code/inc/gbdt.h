#ifndef __GBDT_H__
#define __GBDT_H__

#include <iostream>
#include <stdlib.h>

#include "utilis.h"

namespace ert{
	
class gbdt{

public:
	gbdt(int tree_depth, int feature_pool_size, int split_feature_num, float learning_rate, float lamda);
	gbdt();
private:
	int left_child(int index);
	
	int right_child(int index);
	
	split_node split_tree(const std::vector<train_sample> &samples, int start, int end, const cv::Mat_<float> &feature_pool_coordinate, 
		const cv::Mat_<float> &parent_sum, cv::Mat_<float> &left_child_sum, cv::Mat_<float> &right_child_sum);
public:	
	void generate_tree(std::vector<train_sample> &samples, 
		cv::Mat_<float> &feature_pool_coordinate);
		
	void test_tree(train_sample &sample);
	
	void show_select_feature(cv::Mat_<uchar> &image, cv::Rect &bounding_box, cv::Mat_<float> &feature_pool_coordinate);
	
	void save_model(std::ifstream &fout);
	
	void load_model(std::ifstream &fin);
	
	void print_tree();
private:
	int tree_depth;
	std::vector<split_node> tree_node_split_feature;
	std::vector<cv::Mat_<float> > tree_leaf_regression_value;
	float learning_rate;
	int feature_pool_size;
	int split_feature_num;
	float lamda;
};

}


#endif