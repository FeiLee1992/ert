#ifndef __REGRESSION_H__
#define __REGRESSION_H__


#include "gbdt.h"
#include "utilis.h"

namespace ert{
	
class regression{
public:
	regression(int regressor_size, int cascade_num, int feature_pool_size, int oversample_num, float padding);
	
public:
	void train(std::vector<cv::Mat_<uchar> > &images, std::vector<cv::Rect> &bounding_box, std::vector<cv::Mat_<float> > &ground_truth_shapes,
		int tree_depth, int split_feature_num, float learning_rate, float lamda);
		
	void predict(cv::Mat_<uchar> &image, cv::Mat_<float> &shape, cv::Rect &bounding_box, cv::Mat_<float> &ground_truth_shape);
	
	void predict(cv::Mat_<uchar> &image, cv::Mat_<float> &shape, cv::Rect &bounding_box);
	
	void validate_similarity_transform(cv::Rect &bounding_box, cv::Mat_<float> &shape);
	
	void save_model(std::ifstream &fout);
	
	void load_model(std::ifstream &fin);
private:
	void extract_feature_value(const std::vector<cv::Mat_<uchar> > &images, std::vector<train_sample> &samples, const cv::Mat_<float> &feature_pool_coordinate, const std::vector<int> &index, const cv::Mat_<float> &delta, const cv::Mat_<float> &initialize_shape);
	
	void generate_train_samples(std::vector<cv::Rect> &bounding_box, std::vector<cv::Mat_<float> > &ground_truth_shapes, 
		cv::Mat_<float> &mean_shape, std::vector<train_sample> &train_samples);

	void generate_pixel_feature(const cv::Mat_<float> &initialize_shape, cv::Mat_<float> &feature_pool_coordinate);
	
	void generate_pixel_delta(const cv::Mat_<float> &initialize_shape, const cv::Mat_<float> &feature_pool_coordinate, std::vector<int> &index, cv::Mat_<float> &delta);

	void show_random_feature(cv::Mat_<uchar> &image, cv::Mat_<float> feature_pool_coordinate, cv::Rect &bounding_box);
	
	void reportError(std::vector<train_sample> &train_samples);
	
	void show_train_samples(std::vector<train_sample> &train_samples, std::vector<cv::Mat_<uchar> > &images, int num, bool post_process);
private:
	std::vector<std::vector<int> > indexes;
	std::vector<cv::Mat_<float> > deltas;
	std::vector<cv::Mat_<float> > feature_pool_coordinates;
	std::vector<std::vector<gbdt> > regressors;
	int regressor_size;
	int oversample_num;
	int feature_pool_size;
	int cascade_num;
	float padding;
	cv::Mat_<float> mean_shape;
};
	
	
}


#endif