#include "regression.h"

namespace ert{

	regression::regression(int regressor_size, int cascade_num, int feature_pool_size, int oversample_num, float padding)
	{
		this->regressor_size = regressor_size;
		this->cascade_num = cascade_num;
		this->feature_pool_size = feature_pool_size;
		this->oversample_num = oversample_num;
		this->padding = padding;
		std::cout << "setting: " << std::endl 
			<< "regressor size: " << this->regressor_size << std::endl
			<< "cascade_num: " << this->cascade_num << std::endl
			<< "feature_pool_size: " << this->feature_pool_size << std::endl
			<< "oversample_num: " << this->oversample_num << std::endl
			<< "padding: " << this->padding << std::endl;
	}

	void regression::show_random_feature(cv::Mat_<uchar> &image, cv::Mat_<float> feature_pool_coordinate, cv::Rect &bounding_box)
	{
		train_sample sample;
		sample.rect = bounding_box;
		static int cnt = 0;
		char buf[128];
		cv::Mat_<float> rect_scale_rotate, rect_transform;
		unnormalization(sample, rect_scale_rotate, rect_transform);
		for(size_t i = 0; i < feature_pool_coordinate.rows; i++)
		{
			cv::Mat_<float> tmp(1, 2);
			tmp(0, 0) = feature_pool_coordinate(i, 0);
			tmp(0, 1) = feature_pool_coordinate(i, 1);
			tmp = (rect_scale_rotate * tmp.t()).t() + rect_transform;
			//sample.cur_pts(i, 0) = tmp(0, 0);
			//sample.cur_pts(i, 1) = tmp(0, 1);
			cv::circle(image, cv::Point(tmp(0, 0), tmp(0, 1)), 2, cv::Scalar(255, 255, 255), -1);
		}
		for(size_t i = 0; i < mean_shape.rows; i++)
		{
			cv::Mat_<float> tmp(1, 2);
			tmp(0, 0) = mean_shape(i, 0);
			tmp(0, 1) = mean_shape(i, 1);
			tmp = (rect_scale_rotate * tmp.t()).t() + rect_transform;
			//sample.cur_pts(i, 0) = tmp(0, 0);
			//sample.cur_pts(i, 1) = tmp(0, 1);
			cv::circle(image, cv::Point(tmp(0, 0), tmp(0, 1)), 4, cv::Scalar(0, 0, 0), -1);
		}

		memset(buf, 0, sizeof(buf));
		sprintf(buf, "./%d_feature.jpg", cnt++);
		imwrite(buf, image);
	}
	void regression::reportError(std::vector<train_sample> &train_samples){
		float error = 0;
		for(size_t k = 0; k < train_samples.size(); k++)
		{
			//if(k == 0)
			//	std::cout << train_samples[k].cur_pts << std::endl;
			//cv::Mat_<float> diff = train_samples[k].cur_pts - train_samples[k].target_pts;
			cv::Mat_<float> rect_scale_rotate, rect_transform;
			cv::Mat_<float> cur_pts = (cv::Mat_<float>(train_samples[k].cur_pts.rows, train_samples[k].cur_pts.cols)).
				zeros(train_samples[k].cur_pts.rows, train_samples[k].cur_pts.cols);
			cv::Mat_<float> target_pts = (cv::Mat_<float>(train_samples[k].cur_pts.rows, train_samples[k].cur_pts.cols)).
				zeros(train_samples[k].cur_pts.rows, train_samples[k].cur_pts.cols);
			unnormalization(train_samples[k], rect_scale_rotate, rect_transform);
			for(size_t i = 0; i < target_pts.rows; i++)
			{
				cv::Mat_<float> tmp(1, 2);
				tmp(0, 0) = train_samples[k].target_pts(i, 0);
				tmp(0, 1) = train_samples[k].target_pts(i, 1);
				tmp = (rect_scale_rotate * tmp.t()).t() + rect_transform;
				target_pts(i, 0) = tmp(0, 0);
				target_pts(i, 1) = tmp(0, 1);
				
				tmp(0, 0) = train_samples[k].cur_pts(i, 0);
				tmp(0, 1) = train_samples[k].cur_pts(i, 1);
				tmp = (rect_scale_rotate * tmp.t()).t() + rect_transform;
				cur_pts(i, 0) = tmp(0, 0);
				cur_pts(i, 1) = tmp(0, 1);
				//cv::circle(image, cv::Point(tmp(0, 0), tmp(0, 1)), 2, cv::Scalar(255, 255, 255), -1);
			}
			
			cv::Mat_<float> diff = target_pts - cur_pts;
			int landmark_num = diff.rows;
			int ll_bound = 36;
			int lr_bound = 41;
			int rl_bound = 42;
			int rr_bound = 47;
			if(landmark_num == 5)
			{
				ll_bound = 0;
				lr_bound = 0;
				rl_bound = 1;
				rr_bound = 1;
			}
			//float tmp = std::sqrt(diff.dot(diff));
			float tmp = 0;
			for(size_t err = 0; err < diff.rows; err++)
				tmp += std::sqrt(std::pow(diff(err, 0), 2) + std::pow(diff(err, 1), 2));
			float ocular_dist = 0;
			float lx = 0, ly = 0, rx = 0, ry = 0;
			for(int m = ll_bound; m <= lr_bound; m++)
			{
				lx += target_pts(m, 0);
				ly += target_pts(m, 1);
			}
			for(int m = rl_bound; m <= rr_bound; m++)
			{
				rx += target_pts(m, 0);
				ry += target_pts(m, 1);
			}
			lx /= (1 + lr_bound - ll_bound);
			ly /= (1 + lr_bound - ll_bound);
			rx /= (1 + lr_bound - ll_bound);
			ry /= (1 + lr_bound - ll_bound);
			ocular_dist = std::sqrt(std::pow(lx - rx, 2) + std::pow(ly - ry, 2));
			error += tmp / ocular_dist / train_samples[k].target_pts.rows;
		}	
		std::cout << "report error: " << error / train_samples.size() << std::endl;
	
	}
	void regression::train(std::vector<cv::Mat_<uchar> > &images, std::vector<cv::Rect> &bounding_box, std::vector<cv::Mat_<float> > &ground_truth_shapes,
		int tree_depth, int split_feature_num, float learning_rate, float lamda)
	{
		regressors.resize(cascade_num);
		feature_pool_coordinates.resize(cascade_num);
		std::vector<train_sample> train_samples;
		std::cout << "data augment..." << std::endl;
		generate_train_samples(bounding_box, ground_truth_shapes, mean_shape, train_samples);
		//cv::Mat_<float> tmpl = mean_shape.rowRange(0, 5).t();
		//std::cout << tmpl.rowRange(0, 1) << std::endl;
		//show_train_samples(train_samples, images, images.size(), false);
		//std::cout << "fetch mean shape: " << std::endl << mean_shape << std::endl;
		std::cout << "finish augmenting samples, samples num: " << train_samples.size() << std::endl;
		for(int i = 0; i < cascade_num; i++){
			generate_pixel_feature(mean_shape, feature_pool_coordinates[i]);
			//std::cout << feature_pool_coordinates[i].t() << std::endl;
			//std::cout << std::endl;
		}
		//for(size_t i = 0; i < images.size(); i++)
		//	show_random_feature(images[i], feature_pool_coordinates[0], bounding_box[i]);
		//std::vector<int> index;
		//cv::Mat_<float> delta;
		indexes.resize(cascade_num);
		deltas.resize(cascade_num);
		cv::Mat_<uchar> drawImage;
		images[0].copyTo(drawImage);
		std::cout << "prepare to generate regressor" << std::endl;
		reportError(train_samples);
		for(int i = 0; i < cascade_num; i++)
		{
			//cv::Mat_<float> sum_err = (cv::Mat_<float>(train_samples[0].cur_pts.rows, train_samples[0].cur_pts.cols)).zeros
			//	(train_samples[0].cur_pts.rows, train_samples[0].cur_pts.cols);
			//for(size_t sample_cnt = 0; sample_cnt < train_samples.size(); sample_cnt++)
			//	sum_err += train_samples[sample_cnt].target_pts - train_samples[sample_cnt].cur_pts;
			//std::cout << "cascade regressor: " << i << std::endl;
			//std::cout << "current error: " << std::endl << sum_err.rowRange(0, 5).t() << std::endl;
			//regressors[i].resize(regressor_size);
			//std::cout << ((train_samples[0].target_pts.rowRange(0, 5) - train_samples[0].cur_pts.rowRange(0, 5)).t()) << std::endl;
			generate_pixel_delta(mean_shape, feature_pool_coordinates[i], indexes[i], deltas[i]);
			//std::cout << "begin to extract feature..." << std::endl;
			//std::cout << feature_pool_coordinates[i].rowRange(0, 10).t() << std::endl;
			extract_feature_value(images, train_samples, feature_pool_coordinates[i], indexes[i], deltas[i], mean_shape);
			//for(int k = 0; k < train_samples.size(); k++)
			//std::cout << train_samples[k].feature_values.rowRange(30, 60).t() << std::endl;
			//std::cout << train_samples[6].target_pts.t() << std::endl << train_samples[6].cur_pts.t() << std::endl;
			//std::cout << "extract features finished, go to train gbdt" << std::endl;
			
			int ten_fold = regressor_size / 10;
			
			for(int m = 0; m < regressor_size; m++)
			{
				//if(m == 0)
				//	std::cout << "generate trees for session: " << i << std::endl;
				if(ten_fold != 0 && m % ten_fold == 0)
					std::cout << "train " << m / (float)regressor_size * 100 << "% trees" << std::endl;
				gbdt tree(tree_depth, feature_pool_size, split_feature_num, learning_rate, lamda);
				tree.generate_tree(train_samples, feature_pool_coordinates[i]);
				//if(m == 0)
				//	tree.print_tree();
				regressors[i].push_back(tree);
				//tree.show_select_feature(drawImage, bounding_box[0], feature_pool_coordinates[0]);
			}
			std::cout << "regression stage: " << i << " ";
			reportError(train_samples);
			//std::cout << train_samples[0].target_pts.rowRange(0, 5).t() - train_samples[0].cur_pts.rowRange(0, 5).t() << std::endl;
			//std::cout << std::endl;
			//sum_err.zeros(train_samples[0].cur_pts.rows, train_samples[0].cur_pts.cols);
			//for(size_t sample_cnt = 0; sample_cnt < train_samples.size(); sample_cnt++)
			//	sum_err += train_samples[sample_cnt].target_pts - train_samples[sample_cnt].cur_pts;
			//std::cout << "after cascade error sum: " << std::endl;
			//std::cout << (sum_err.rowRange(0, 8).t()) << std::endl;
			//std::cout << train_samples[0].cur_pts.rowRange(0, 8).t() << std::endl;

		}
		//cv::imwrite("./select_feature.jpg", drawImage);
		//show_train_samples(train_samples, images, images.size(), true);
	}
	
	void regression::show_train_samples(std::vector<train_sample> &train_samples, std::vector<cv::Mat_<uchar> > &images, int num, bool post_process)
	{
		std::vector<int> visit_flag(images.size());
		
		for(size_t i = 0; i < train_samples.size(); i++)
		{
			int index = train_samples[i].index;
			if(visit_flag[index])
				continue;
			visit_flag[index] = 1;
			cv::Mat_<float> rect_scale_rotate, rect_transform;
			unnormalization(train_samples[i], rect_scale_rotate, rect_transform);
			for(size_t k = 0; k < train_samples[i].cur_pts.rows; k++)
			{
				cv::Mat_<float> tmp(1, 2);
				tmp(0, 0) = train_samples[i].cur_pts(k, 0);
				tmp(0, 1) = train_samples[i].cur_pts(k, 1);
				tmp = (rect_scale_rotate * tmp.t()).t() + rect_transform;
				cv::circle(images[train_samples[i].index], cv::Point((int)tmp(0, 0), (int)tmp(0, 1)), 2, cv::Scalar(255, 255, 255), -1);
			}
			char buf[32];
			memset(buf, 0, sizeof(buf));
			//cv::Mat_<float> tmp = train_samples[i].target_pts.rowRange(0, 8).t() - train_samples[i].cur_pts.rowRange(0, 8).t();
			//std::cout << tmp.rowRange(0, 1) << std::endl;
			sprintf(buf, "./result/%d.jpg", index);
			cv::imwrite(buf, images[train_samples[i].index]);
				
		}
	}
	
	void regression::validate_similarity_transform(cv::Rect &bounding_box, cv::Mat_<float> &shape)
	{
		train_sample sample;
		sample.target_pts = shape;
		sample.rect = bounding_box;
		sample.cur_pts = normalization(sample);
		std::cout << sample.cur_pts.rowRange(0, 5).t() << std::endl;
		cv::Mat_<float> rect_scale_rotate, rect_transform;
		unnormalization(sample, rect_scale_rotate, rect_transform);
		for(size_t i = 0; i < sample.cur_pts.rows; i++)
		{
			cv::Mat_<float> tmp(1, 2);
			tmp(0, 0) = sample.cur_pts(i, 0);
			tmp(0, 1) = sample.cur_pts(i, 1);
			tmp = (rect_scale_rotate * tmp.t()).t() + rect_transform;
			sample.cur_pts(i, 0) = tmp(0, 0);
			sample.cur_pts(i, 1) = tmp(0, 1);
		}
		shape = sample.cur_pts;
	}	
	
	
	
	
	void regression::predict(cv::Mat_<uchar> &image, cv::Mat_<float> &shape, cv::Rect &bounding_box)
	{
		train_sample sample;
		mean_shape.copyTo(sample.cur_pts);
		//std::cout << mean_shape << std::endl;
		sample.rect = bounding_box;
		for(int i = 0; i < cascade_num; i++)
		{
			//std::cout << "predict level: " << i << std::endl;
			cv::Mat_<float> scale_rotate, transform;
			sample.feature_values = (cv::Mat_<int>(feature_pool_coordinates[i].rows, 1)).zeros(feature_pool_coordinates[i].rows, 1);
			get_similarity_transform(mean_shape, sample.cur_pts, scale_rotate, transform);
			//std::cout << scale_rotate << std::endl;
			//std::cout << transform << std::endl;
			cv::Mat_<float> rect_scale_rotate, rect_transform;
			unnormalization(sample, rect_scale_rotate, rect_transform);
			for(size_t m = 0; m < feature_pool_coordinates[i].rows; m++)
			{
				cv::Mat_<float> tmp(1, 2);
				tmp(0, 0) = deltas[i](m, 0);
				tmp(0, 1) = deltas[i](m, 1);
				tmp = (scale_rotate * tmp.t()).t() + transform;
				tmp(0, 0) += sample.cur_pts(indexes[i][m], 0);
				tmp(0, 1) += sample.cur_pts(indexes[i][m], 1);
				tmp = (rect_scale_rotate * tmp.t()).t() + rect_transform;
				if(tmp(0, 1) < 0 || tmp(0, 1) > image.rows ||
					tmp(0, 0) < 0 || tmp(0, 0) > image.cols)
					sample.feature_values(m, 0) = 0;
				else
					sample.feature_values(m, 0) = (int)image.at<uchar>((int)tmp(0, 1), (int)tmp(0, 0));
			}
			//std::cout << sample.feature_values << std::endl;
			for(int m = 0; m < regressor_size; m++){
				regressors[i][m].test_tree(sample);
				//if(m == 0)
				//	regressors[i][m].print_tree();
			}
		}
		cv::Mat_<float> rect_scale_rotate, rect_transform;
		unnormalization(sample, rect_scale_rotate, rect_transform);
		//cv::Mat_<float> tmp = mean_shape.rowRange(0, 5).t() - sample.cur_pts.rowRange(0, 5).t();
		//std::cout << tmp.rowRange(0, 1) << std::endl;
		for(size_t i = 0; i < sample.cur_pts.rows; i++)
		{
			cv::Mat_<float> tmp(1, 2);
			tmp(0, 0) = sample.cur_pts(i, 0);
			tmp(0, 1) = sample.cur_pts(i, 1);
			tmp = (rect_scale_rotate * tmp.t()).t() + rect_transform;
			sample.cur_pts(i, 0) = tmp(0, 0);
			sample.cur_pts(i, 1) = tmp(0, 1);
		}

		//std::cout << sample.rect << std::endl;
		sample.cur_pts.copyTo(shape);
	}
	void regression::predict(cv::Mat_<uchar> &image, cv::Mat_<float> &shape, cv::Rect &bounding_box, cv::Mat_<float> &ground_truth_shape)
	{
		train_sample sample;
		mean_shape.copyTo(sample.cur_pts);

		sample.target_pts = ground_truth_shape;
		//std::cout << mean_shape << std::endl;
		sample.rect = bounding_box;
		sample.target_pts = normalization(sample);

		for(int i = 0; i < 1; i++)
		{
			cv::Mat_<float> scale_rotate, transform;
			sample.feature_values = (cv::Mat_<int>(feature_pool_coordinates[i].rows, 1)).zeros(feature_pool_coordinates[i].rows, 1);
			get_similarity_transform(mean_shape, sample.cur_pts, scale_rotate, transform);
			cv::Mat_<float> rect_scale_rotate, rect_transform;
			unnormalization(sample, rect_scale_rotate, rect_transform);
			for(size_t m = 0; m < feature_pool_coordinates[i].rows; m++)
			{
				cv::Mat_<float> tmp(1, 2);
				tmp(0, 0) = deltas[i](m, 0);
				tmp(0, 1) = deltas[i](m, 1);
				tmp = (scale_rotate * tmp.t()).t() + transform;
				tmp(0, 0) += sample.cur_pts(indexes[i][m], 0);
				tmp(0, 1) += sample.cur_pts(indexes[i][m], 1);
				tmp = (rect_scale_rotate * tmp.t()).t() + rect_transform;
				if(tmp(0, 1) < 0 || tmp(0, 1) > image.rows ||
					tmp(0, 0) < 0 || tmp(0, 0) > image.cols)
					sample.feature_values(m, 0) = 0;
				else
					sample.feature_values(m, 0) = (int)image.at<uchar>((int)tmp(0, 1), (int)tmp(0, 0));
			}
			for(int m = 0; m < regressor_size; m++)
				regressors[i][m].test_tree(sample);
		}
		//std::cout << sample.cur_pts.rowRange(0, 5).t() - sample.target_pts.rowRange(0, 5).t() << std::endl;

		//cv::Mat_<float> result = sample.cur_pts - sample.target_pts;
		//std::cout << result.rowRange(0, 8).t() << std::endl;
		cv::Mat_<float> rect_scale_rotate, rect_transform;
		unnormalization(sample, rect_scale_rotate, rect_transform);
		for(size_t i = 0; i < sample.cur_pts.rows; i++)
		{
			cv::Mat_<float> tmp(1, 2);
			tmp(0, 0) = sample.cur_pts(i, 0);
			tmp(0, 1) = sample.cur_pts(i, 1);
			tmp = (rect_scale_rotate * tmp.t()).t() + rect_transform;
			sample.cur_pts(i, 0) = tmp(0, 0);
			sample.cur_pts(i, 1) = tmp(0, 1);
		}
		cv::Mat_<float> diff = sample.cur_pts - ground_truth_shape;
		/*float dist = 0;
		for(int i = 0; i < diff.rows; i++)
			dist += std::sqrt(std::pow(diff(i, 0), 2) + std::pow(diff(i, 1), 2));
		int lx = 0, ly = 0, rx = 0, ry = 0;
		for(int i = 36; i <= 41; i++)
		{
			lx += ground_truth_shape(i, 0);
			ly += ground_truth_shape(i, 1);
		}
		for(int i = 42; i <= 47; i++)
		{
			rx += ground_truth_shape(i, 0);
			ry += ground_truth_shape(i, 1);
		}
		lx /= 6;
		ly /= 6;
		rx /= 6;
		ry /= 6;
		float inter_ocular = std::sqrt(std::pow(lx - rx, 2) + std::pow(ly - ry, 2));
		float err = dist / inter_ocular;*/
		//std::cout << err << std::endl;
		//std::cout << sample.cur_pts.rowRange(0, 5).t() - ground_truth_shape.rowRange(0, 5).t() << std::endl;
		//std::cout << sample.rect << std::endl;
		shape = sample.cur_pts;
	}
	void regression::extract_feature_value(const std::vector<cv::Mat_<uchar> > &images, std::vector<train_sample> &samples, 
		const cv::Mat_<float> &feature_pool_coordinate, const std::vector<int> &index, 
		const cv::Mat_<float> &delta, const cv::Mat_<float> &initialize_shape)
	{
		
		//std::cout << "extract features for " << samples.size() << " samples, feature pool size: " << feature_pool_coordinate.rows << std::endl;
		for(size_t i = 0; i < samples.size(); i++)
		{
			cv::Mat_<float> scale_rotate, transform;
			samples[i].feature_values = (cv::Mat_<int>(feature_pool_coordinate.rows, 1)).zeros(feature_pool_coordinate.rows, 1);
			//std::cout << initialize_shape << std::endl << samples[i].cur_pts << std::endl;
			
			get_similarity_transform(initialize_shape, samples[i].cur_pts, scale_rotate, transform);
			cv::Mat_<float> rect_scale_rotate, rect_transform;
			unnormalization(samples[i], rect_scale_rotate, rect_transform);

			for(size_t m = 0; m < feature_pool_coordinate.rows; m++)
			{
				cv::Mat_<float> tmp(1, 2);
				tmp(0, 0) = delta(m, 0);
				tmp(0, 1) = delta(m, 1);
				tmp = (scale_rotate * tmp.t()).t() + transform;
				tmp(0, 0) += samples[i].cur_pts(index[m], 0);
				tmp(0, 1) += samples[i].cur_pts(index[m], 1);
				tmp = (rect_scale_rotate * tmp.t()).t() + rect_transform;
				//std::cout << tmp(0, 0) << ", " << tmp(0, 1) << std::endl;
				if(tmp(0, 1) < 0 || tmp(0, 1) > images[samples[i].index].rows ||
					tmp(0, 0) < 0 || tmp(0, 0) > images[samples[i].index].cols)
					samples[i].feature_values(m, 0) = 0;
				else
					samples[i].feature_values(m, 0) = (int)images[samples[i].index].at<uchar>((int)tmp(0, 1), (int)tmp(0, 0));
			//	if(i == 7 * oversample_num && m % 5 == 0)
			//	{
			//		cv::circle(images[7], cv::Point((int)tmp(0, 0), (int)tmp(0, 1)), 2, cv::Scalar(255, 255, 255), -1);
			//		cv::rectangle(images[7], samples[i].rect, cv::Scalar(255, 255, 255), 1, 1, 0);
			//		//char buf[12];
			//		//memset(buf, 0, sizeof(buf));
			//		//sprintf(buf, "%d", samples[i].feature_values(m, 0));
			//		//cv::putText(images[0], buf, cv::Point((int)tmp(0, 0), (int)tmp(0, 1)), CV_FONT_HERSHEY_DUPLEX, 0.3, cv::Scalar(255, 255, 255));
			//	}		
			}
			//if(i == 7 * oversample_num)
			//	cv::imwrite("./extract_feature_pos.jpg", images[7]);
		}
	}
	
	void regression::generate_train_samples(std::vector<cv::Rect> &bounding_box, std::vector<cv::Mat_<float>> &ground_truth_shapes, 
		cv::Mat_<float> &mean_shape, std::vector<train_sample> &train_samples)
	{
		mean_shape = (cv::Mat_<float>(ground_truth_shapes[0].rows, ground_truth_shapes[0].cols)).zeros(ground_truth_shapes[0].rows, ground_truth_shapes[0].cols);
		int cnt = 0;
		train_samples.resize(oversample_num * bounding_box.size());
		//int ground_true_shape_cnt = ground_truth_shapes.size() / 3;
		for(size_t i = 0; i < bounding_box.size(); i++)
		{
			train_sample sample;
			sample.target_pts = ground_truth_shapes[i];
			sample.rect = bounding_box[i];
			
			sample.index = i;
			sample.target_pts = normalization(sample);
			//std::cout << "generate sample" << std::endl;
			for(int j = 0; j < oversample_num; j++)
				copy(train_samples[i * oversample_num + j], sample);
			//if(i < ground_true_shape_cnt){
				mean_shape += sample.target_pts;
				cnt++;
			//}
		}
		//std::cout << "ground true images cnt: " << ground_true_shape_cnt << " images used for mean_shape cnt: " << cnt << std::endl;
		//std::cout << mean_shape << std::endl;
		mean_shape /= cnt;
		for(size_t i = 0; i < train_samples.size(); i++)
		{
			if(i % oversample_num == 0){
				mean_shape.copyTo(train_samples[i].cur_pts);
			}
			else{
				int left, right;
				do{
					left = std::rand() % train_samples.size();
					right = std::rand() % train_samples.size();
				}while(train_samples[left].index == train_samples[i].index && train_samples[right].index == train_samples[i].index);
				float alpha = std::rand() / (RAND_MAX + 1.0);
				cv::Mat_<float> tmp = (alpha + 0.001) * train_samples[left].target_pts + (0.999 - alpha) * train_samples[right].target_pts;
				tmp.copyTo(train_samples[i].cur_pts);
				//train_samples[i].cur_pts = mean_shape;
			}
		}
	}

	void regression::generate_pixel_feature(const cv::Mat_<float> &initialize_shape, cv::Mat_<float> &feature_pool_coordinate)
	{
		float min_x = std::numeric_limits<float>::max(), min_y = std::numeric_limits<float>::max();
		float max_x = 0, max_y = 0;
		feature_pool_coordinate = cv::Mat_<float>(feature_pool_size, 2);
		for(size_t i = 0; i < initialize_shape.rows; i++)
		{
			if(min_x > initialize_shape(i, 0))
				min_x = initialize_shape(i, 0);
			if(max_x < initialize_shape(i, 0))
				max_x = initialize_shape(i, 0);
			if(min_y > initialize_shape(i, 1))
				min_y = initialize_shape(i, 1);
			if(max_y < initialize_shape(i, 1))
				max_y = initialize_shape(i, 1);
		}
		//std::cout << max_x << ", " << min_x << ", " << max_y << ", " << min_y << std::endl;
		min_x -= padding;
		min_y -= padding;
		max_x += padding;
		max_y += padding;
		for(int i = 0; i < feature_pool_size; i++)
		{
			float x = 0, y = 0;
			//do{
				x = ((std::rand() / (float)(RAND_MAX)) * (max_x - min_x)) + min_x;
				y = ((std::rand() / (float)(RAND_MAX)) * (max_y - min_y)) + min_y;
			//}while(x < 0.1 || x > 0.9 || y < 0.1 || y > 0.9);
			feature_pool_coordinate(i, 0) = x;
			feature_pool_coordinate(i, 1) = y;
		}
	}
	
	void regression::generate_pixel_delta(const cv::Mat_<float> &initialize_shape, const cv::Mat_<float> &feature_pool_coordinate, std::vector<int> &index, cv::Mat_<float> &delta)
	{
		delta = cv::Mat_<float>(feature_pool_size, 2);
		index.resize(feature_pool_size);
		for(size_t i = 0; i < feature_pool_coordinate.rows; i++)
		{
			float x = feature_pool_coordinate(i, 0);
			float y = feature_pool_coordinate(i, 1);
			float min_dist = std::numeric_limits<float>::max();
			int nearest_index = 0;
			for(size_t m = 0; m < initialize_shape.rows; m++)
			{
				float _ref_x = initialize_shape(m, 0);
				float _ref_y = initialize_shape(m, 1);
				float dist = std::sqrt(std::pow(x - _ref_x, 2) + std::pow(y - _ref_y, 2));
				if(dist < min_dist)
				{
					min_dist = dist;
					nearest_index = m;
				}
			}
			index[i] = nearest_index;
			delta(i, 0) = x - initialize_shape(nearest_index, 0);
			delta(i, 1) = y - initialize_shape(nearest_index, 1);
		}
	}
	
	std::vector<std::vector<int> > indexes;
	std::vector<cv::Mat_<float> > deltas;
	std::vector<cv::Mat_<float> > feature_pool_coordinates;
	std::vector<std::vector<gbdt> > regressors;
	int regressor_size;
	int oversample_num;
	int feature_pool_size;
	int cascade_num;
	cv::Mat_<float> mean_shape;
	
	
	/*void regression::load_model(ifstream &fin)
	{
		fin >> feature_pool_size;
		fin >> oversample_num;
		fin >> cascade_num;
		fin >> regressor_size;
		int landmark_num;
		fin >> landmark_num;
		mean_shape = cv::Mat_<float>(landmark_num, 2);
		for(int i = 0; i < landmark_num; i++)
			fin >> mean_shape(i, 0) >> mean_shape(i, 1);
		regressors.resize(cascade_num);
		for(int i = 0; i < cascade_num; i++)
		{
			gbdt tree;
			tree.load_model(fin);
			regressors.push_back(tree);
		}
	}
	
	void regression::save_model(ifstream &fout)
	{
		fout << feature_pool_size;
		fout << oversample_num;
		fout << cascade_num;
		fout << regressor_size;
		int landmark_num;
		fout << landmark_num;
		mean_shape = cv::Mat_<float>(landmark_num, 2);
		for(int i = 0; i < landmark_num; i++)
			fout << mean_shape(i, 0) << mean_shape(i, 1);
		regressors.resize(cascade_num);
		for(int i = 0; i < cascade_num; i++)
		{
			for(int j = 0; j < regressor_size; j++)
				regressors[i][j].save_model(fin);
		}
	}*/
}