#include <omp.h>
#include "gbdt.h"

namespace ert{
	gbdt::gbdt(int tree_depth, int feature_pool_size, int split_feature_num, float learning_rate, float lamda)
	{
		this->tree_depth = tree_depth;
		this->feature_pool_size = feature_pool_size;
		this->split_feature_num = split_feature_num;
		this->learning_rate = learning_rate;
		this->lamda = lamda;
	}
	gbdt::gbdt(){}

	int gbdt::left_child(int index)
	{
		return index * 2 + 1;
	}
	
	int gbdt::right_child(int index)
	{
		return index * 2 + 2;
	}
	
	split_node gbdt::split_tree(const std::vector<train_sample> &samples, int start, int end, const cv::Mat_<float> &feature_pool_coordinate, 
		const cv::Mat_<float> &parent_sum, cv::Mat_<float> &left_child_sum, cv::Mat_<float> &right_child_sum)
	{
		//left_child_sum.zeros(samples[0].target_pts.rows, samples[0].target_pts.cols);
		//right_child_sum.zeros(samples[0].target_pts.rows, samples[0].target_pts.cols);
		std::vector<split_node> generate_split_feature;
		float alpha;
		float accept_prob = 0;
		split_node _sp;
		for(int i = 0; i < split_feature_num; i++)
		{
			do{
				_sp.x = std::rand() % feature_pool_size;
				_sp.y = std::rand() % feature_pool_size;
				float dist = std::sqrt(std::pow(feature_pool_coordinate(_sp.x, 0) - feature_pool_coordinate(_sp.y, 0), 2) +
					std::pow(feature_pool_coordinate(_sp.x, 1) - feature_pool_coordinate(_sp.y, 1), 2)); 
                accept_prob = std::exp(-dist/lamda);
				alpha = std::rand() / (RAND_MAX + 1.0);

			}while(_sp.x == _sp.y || !(accept_prob > alpha));
			_sp.threshold = ((std::rand() / (RAND_MAX + 1.0) * std::numeric_limits<uchar>::max()) - 128) / 2;
			generate_split_feature.push_back(_sp);
		}
		//int positive_cnt = 0;
		//for(int i = 0; i < generate_split_feature.size(); i++)
		//{
		//	std::cout << generate_split_feature[i].threshold << " ";
		//	positive_cnt += generate_split_feature[i].threshold > 0 ? 1 : 0;
		//}
		//std::cout << "positive_cnt: " << positive_cnt << std::endl;
		std::vector<cv::Mat_<float> > sum_left(split_feature_num);
		std::vector<int> left_cnt(split_feature_num);
		//int *left_cnt = new int[split_feature_num];
		//memset(left_cnt, 0, sizeof(int) * split_feature_num);
		for(int m = 0; m < split_feature_num; m++)
		{
			sum_left[m] = (cv::Mat_<float>(parent_sum.rows, parent_sum.cols)).zeros(parent_sum.rows, parent_sum.cols);
			//std::cout << sum_left[m].rowRange(0, 5).t() << std::endl;
			//#pragma omp parallel for
			for(int i = start; i < end; i++)
			{
				if((samples[i].feature_values(generate_split_feature[m].x, 0) - samples[i].feature_values(generate_split_feature[m].y, 0)) > generate_split_feature[m].threshold)
				{
					left_cnt[m]++;
					sum_left[m] += (samples[i].target_pts - samples[i].cur_pts);
				}
			}
		}
		//for(int i = 0; i < split_feature_num; i++)
		//	std::cout << left_cnt[i] << ", " << generate_split_feature[i].threshold << ";";
		//std::cout << std::endl;
		float best_score = -1;
		int best_feat = 0;
		//int total_left_index = -1;
		for(int m = 0; m < split_feature_num; m++)
		{
			float score = -1;
            unsigned long right_cnt = end - start - left_cnt[m];
            if (left_cnt[m] != 0 && right_cnt != 0)
            {
                cv::Mat_<float> temp = parent_sum - sum_left[m];
                score = sum_left[m].dot(sum_left[m]) / left_cnt[m] + temp.dot(temp) / right_cnt;
				//std::cout << score << "-> " << right_cnt << " : " << left_cnt[m] << std::endl;
                if (score > best_score)
                {
                    best_score = score;
                    best_feat = m;
                }
            }
			//else if(right_cnt == 0)
			//	total_left_index = m;
		}
		//if(best_feat == -1)
		//	best_feat = total_left_index;
		//std::cout << std::endl;
		sum_left[best_feat].copyTo(left_child_sum);
		if(left_cnt[best_feat] != 0)
		{
			//std::cout << left_cnt[best_feat] << " : " << end - start - left_cnt[best_feat] << " -> (" << generate_split_feature[best_feat].x << ", " << generate_split_feature[best_feat].y << "): " << generate_split_feature[best_feat].threshold << std::endl;
			right_child_sum = parent_sum - left_child_sum;
		}
		else{
			left_child_sum = (cv::Mat_<float>(parent_sum.rows, parent_sum.cols)).zeros(parent_sum.rows, parent_sum.cols);
			parent_sum.copyTo(right_child_sum);
		}
		//static int i = 0;
		//std::cout << i << ": ";
		//cv::Mat_<float> tmp;
		//tmp = parent_sum.rowRange(0, 5).t();
		//std::cout << "parent: " << tmp.rowRange(0, 1) << " num: " << end - start << std::endl;
		//tmp = left_child_sum.rowRange(0, 5).t();
		//std::cout << 2 * i + 1 << ": ";
		//std::cout << "left_child: " << tmp.rowRange(0, 1) << " num: " << left_cnt[best_feat] << std::endl;
		//std::cout << 2 * i + 2 << ": ";i++;
		//tmp = right_child_sum.rowRange(0, 5).t();
		//std::cout << "right_child: " << tmp.rowRange(0, 1) << " num: " << end - start - left_cnt[best_feat] << std::endl;
		//std::cout << "sum: " << parent_sum.rowRange(0, 5).t() - left_child_sum.rowRange(0, 5).t() - right_child_sum.rowRange(0, 5).t() << std::endl;
		//std::cout << std::endl;
		//std::cout << parent_sum.t() - left_child_sum.t() - right_child_sum.t() << std::endl;
		//std::cout << "threshold: " << generate_split_feature[best_feat].threshold << ": " << left_cnt[best_feat] << " || " << end - start - left_cnt[best_feat] << std::endl;
		//delete []left_cnt;
		return generate_split_feature[best_feat];
	}
	

	void gbdt::generate_tree(std::vector<train_sample> &samples, 
		cv::Mat_<float> &feature_pool_coordinate)
	{
		//std::cout << std::endl;
		std::deque<std::pair<int, int> > piecewire_constant;
		piecewire_constant.push_back(std::pair<int, int>(0, samples.size()));
		int split_node_num = std::pow(2, tree_depth) - 1;
		std::vector<cv::Mat_<float>> sum(split_node_num * 2 + 1);
		for(size_t i = 0; i < split_node_num * 2 + 1; i++){
			sum[i] = (cv::Mat_<float>(samples[0].target_pts.rows, samples[0].target_pts.cols)).zeros(samples[0].target_pts.rows, samples[0].target_pts.cols);
			//cv::Mat_<float> tmp = sum[i].rowRange(0, 5).t();
			//std::cout << tmp.rowRange(0, 1) << std::endl;
		}
		//#pragma omp parallel for
		for(size_t i = 0; i < samples.size(); i++){
			sum[0] += (samples[i].target_pts - samples[i].cur_pts);
			//cv::Mat_<float> tmp = (samples[i].target_pts.rowRange(0, 5).t() - samples[i].cur_pts.rowRange(0, 5).t());
			//std::cout << tmp.rowRange(0, 1) << std::endl;
		}
		//std::cout << std::endl;
		//for(size_t i = 0; i < samples.size(); i++)
		//{
		//	cv::Mat_<float> tmp = (samples[i].cur_pts.rowRange(0, 5).t());
		//	std::cout << tmp.rowRange(0, 1) << std::endl;
		//}
		//std::cout << std::endl;
		//cv::Mat_<float> tmp = sum[0].rowRange(0, 5).t();
		//std::cout << tmp.rowRange(0, 1) << std::endl;
		tree_node_split_feature.resize(split_node_num);
		//std::cout << "***************************" << std::endl;
		for(int i = 0; i < split_node_num; i++)
		{
			std::pair<int, int> parts = piecewire_constant.front();
			piecewire_constant.pop_front();
			int m = parts.first;
			//std::cout << "node index: " << i << ": " << std::endl;
			split_node feature = split_tree(samples, parts.first, parts.second, feature_pool_coordinate, sum[i], sum[left_child(i)], sum[right_child(i)]);
			tree_node_split_feature[i] = feature;
			
			//std::cout << "(" << feature.x << ", " << feature.y << ") = " << feature.threshold << " || ";

			//std::cout << std::endl;
			//for(int j = parts.first; j < parts.second; j++)
			//{
			//	cv::Mat_<float> tmp = samples[j].target_pts.rowRange(0, 5).t() - samples[j].cur_pts.rowRange(0, 5).t();
			//	std::cout << tmp.rowRange(0, 1) << ": " << samples[j].feature_values(feature.x, 0) - samples[j].feature_values(feature.y, 0) << " || " << feature.threshold << std::endl;
			//}
		
			//std::cout << std::endl;
			//std::cout << "sort array: " << parts.first << " -> " << parts.second << std::endl;
			for(int j = parts.first; j < parts.second; j++)
			{
				if(samples[j].feature_values(feature.x, 0) - samples[j].feature_values(feature.y, 0) > feature.threshold){
					//samples[m++].swap(samples[j]);
					//if(j != m)
					swap(&samples[j], &samples[m]);
					m++;
				}
			}
			//std::cout << std::endl;

			//for(int j = 0; j < samples.size(); j++)
			//{
			//	cv::Mat_<float> tmp = samples[j].target_pts.rowRange(0, 5).t() - samples[j].cur_pts.rowRange(0, 5).t();
			//	std::cout << tmp.rowRange(0, 1) << ": " << samples[j].feature_values(feature.x, 0) - samples[j].feature_values(feature.y, 0) << " || " << feature.threshold << std::endl;
			//}
			//std::cout << std::endl;
			//std::cout << feature.threshold << ": ";
			//for(int j = parts.first; j < parts.second; j++)
			//{		
			//	std::cout << samples[j].feature_values(feature.x, 0) - samples[j].feature_values(feature.y, 0) << ", ";
			//	if(j == m - 1)
			//		std::cout << " || ";
			//}
			//std::cout << std::endl;
			//std::cout << "-------------->: " << i << std::endl;
			//std::cout << "index: " << i << std::endl;
			
			//for(int k = parts.first; k < m; k++)
			//{
			//	cv::Mat_<float> tmp = (samples[k].target_pts.rowRange(0, 5).t() - samples[k].cur_pts.rowRange(0, 5).t());;
			//	std::cout << tmp.rowRange(0, 1) << ": " << samples[k].feature_values(feature.x, 0) - samples[k].feature_values(feature.y, 0) << std::endl;
			//}
			//std::cout << " || ( " << feature.threshold << " )" << std::endl;
			//for(int k = m; k < parts.second; k++)
			//{
			//	cv::Mat_<float> tmp = (samples[k].target_pts.rowRange(0, 5).t() - samples[k].cur_pts.rowRange(0, 5).t());;
			//	std::cout << tmp.rowRange(0, 1) << ": " << samples[k].feature_values(feature.x, 0) - samples[k].feature_values(feature.y, 0) << std::endl;
			//}
			//cv::Mat_<float> tmp;
			//tmp = sum[i].rowRange(0, 5).t();
			//std::cout << tmp.rowRange(0, 1) << " : " << parts.second - parts.first << std::endl;
			//tmp = sum[left_child(i)].rowRange(0, 5).t();
			//std::cout << tmp.rowRange(0, 1) << " : " << m - parts.first << std::endl;
			//tmp = sum[right_child(i)].rowRange(0, 5).t();
			//std::cout << tmp.rowRange(0, 1) << " : " << parts.second - m << std::endl;
			//std::cout << std::endl;
			piecewire_constant.push_back(std::pair<int, int>(parts.first, m));
			piecewire_constant.push_back(std::pair<int, int>(m, parts.second));
			//std::cout << sum[i].rowRange(0, 5).t() << ": " << parts.second - parts.first << std::endl;
			//std::cout << sum[left_child(i)].rowRange(0, 5).t() << ": " << m - parts.first << std::endl;
			//std::cout << sum[right_child(i)].rowRange(0, 5).t() << ": " << parts.second - m << std::endl;
		}
		//std::cout << "****************************: tree leaf sum: " << std::endl;
		//for(size_t i = 0; i < split_node_num * 2 + 1; i++)
		//{
		//	cv::Mat_<float> tmp;
		//	tmp = sum[i].rowRange(0, 5).t();
		//	if(i >= split_node_num)
		//	std::cout << i << " : " << tmp.rowRange(0, 1) << " : " << piecewire_constant[i - split_node_num].second - piecewire_constant[i - split_node_num].//first << std::endl;
		//	else
		//		std::cout << i << " : " << tmp.rowRange(0, 1) << std::endl;
		//}
		//std::cout << std::endl;
		//for(int i = split_node_num; i < split_node_num * 2 + 1; i++){
		//	cv::Mat_<float> tmp = sum[i].rowRange(0, 5).t();
		//	std::cout << tmp.rowRange(0, 1) << ": " << piecewire_constant[i - split_node_num].second - piecewire_constant[i - split_node_num].first << std::endl;
		//}
		//std::cout << piecewire_constant.size() << std::endl;
		tree_leaf_regression_value.resize(piecewire_constant.size());
		
		//cv::Mat_<float> tmp = (cv::Mat_<float>(sum[0].rows, sum[0].cols)).zeros(sum[0].rows, sum[0].cols);
		//for(int i = 0; i < tree_leaf_regression_value.size(); i++)
		//	tmp += sum[split_node_num + i];
		//std::cout << tmp.rowRange(0, 10).t() << std::endl;
		//std::cout << std::endl;
		//std::cout << sum[1].t() + sum[2].t() - sum[0].t() << std::endl;
		//std::cout << sum[7].t() + sum[8].t() + sum[9].t() + sum[10].t() +
		//	sum[11].t() + sum[12].t() + sum[13].t() + sum[14].t() - sum[0].t()
		//	<< std::endl;
		//tmp = sum[1] + sum[2];
		//std::cout << tmp.rowRange(0, 5).t() << std::endl;
		//std::cout << sum[3].rowRange(0, 5).t() + sum[4].rowRange(0, 5).t() + sum[5].rowRange(0, 5).t() + sum[6].rowRange(0, 5).t()<< std::endl;
		//std::cout << ".";
		//std::cout << "----------------------------------->: tree output: " << std::endl;
		for(size_t i = 0; i < tree_leaf_regression_value.size(); i++)
		{
			tree_leaf_regression_value[i] = (cv::Mat_<float>(samples[0].target_pts.rows, samples[0].target_pts.cols))
				.zeros(samples[0].target_pts.rows, samples[0].target_pts.cols);
			if(piecewire_constant[i].second != piecewire_constant[i].first)
			{
				tree_leaf_regression_value[i] += (sum[split_node_num + i] * learning_rate) / (piecewire_constant[i].second - piecewire_constant[i].first);
				//std::cout << tree_leaf_regression_value[i] << std::endl;
				//std::cout << "update samples in: " << i << std::endl;
				//#pragma omp parallel for
				for(int m = piecewire_constant[i].first; m < piecewire_constant[i].second; m++)
				{
					//cv::Mat_<float> tmp = samples[m].cur_pts.rowRange(0, 5).t();
					//std::cout << tmp.rowRange(0, 1) << std::endl;
					samples[m].cur_pts += tree_leaf_regression_value[i]; 
					//tmp = samples[m].cur_pts.rowRange(0, 5).t();
					//std::cout << tmp.rowRange(0, 1) << std::endl;
				}
				//cv::Mat_<float> tmp = tree_leaf_regression_value[i].rowRange(0, 5).t();
				//std::cout << tmp.rowRange(0, 1) << ": " << (piecewire_constant[i].second - piecewire_constant[i].first) 
				//	<< ", " << piecewire_constant[i].first << ", " << piecewire_constant[i].second << std::endl;
			}
			
		}
		//std::cout << "<-------------------------------->: samples cur_pts: " << std::endl;
		//for(int i = 0; i < samples.size(); i++)
		//{
		//	cv::Mat_<float> tmp = samples[i].cur_pts.rowRange(0, 5).t();
		//	std::cout << tmp.rowRange(0, 1) << std::endl;
		//}
		//std::cout << std::endl;
	}
	void gbdt::test_tree(train_sample &sample)
	{
		int index = 0;
		int split_node_num = std::pow(2, tree_depth) - 1;
		while(index < split_node_num)
		{
			if(sample.feature_values(tree_node_split_feature[index].x, 0) - sample.feature_values(tree_node_split_feature[index].y, 0) > tree_node_split_feature[index].threshold)
				index = index * 2 + 1;
			else
				index = index * 2 + 2;
		}
		//cv::Mat_<float> tmp = tree_leaf_regression_value[index - split_node_num].rowRange(0, 5).t();
		//std::cout << "index: " << index << ", " << tmp.rowRange(0, 1) << std::endl;
		sample.cur_pts += tree_leaf_regression_value[index - split_node_num];
			
	}
	
	void gbdt::show_select_feature(cv::Mat_<uchar> &image, cv::Rect &bounding_box, cv::Mat_<float> &feature_pool_coordinate)
	{
		train_sample sample;
		sample.rect = bounding_box;
		
		cv::Mat_<float> rect_scale_rotate, rect_transform;
		unnormalization(sample, rect_scale_rotate, rect_transform);
		for(size_t i = 0; i < tree_node_split_feature.size(); i++)
		{
			int x = tree_node_split_feature[i].x;
			int y = tree_node_split_feature[i].y;
			cv::Mat_<float> tmp(1, 2);
			tmp(0, 0) = feature_pool_coordinate(x, 0);
			tmp(0, 1) = feature_pool_coordinate(x, 1);
			tmp = (rect_scale_rotate * tmp.t()).t() + rect_transform;
			
			cv::Mat_<float> tmp_(1, 2);
			tmp_(0, 0) = feature_pool_coordinate(y, 0);
			tmp_(0, 1) = feature_pool_coordinate(y, 1);
			tmp_ = (rect_scale_rotate * tmp_.t()).t() + rect_transform;
			cv::line(image, cv::Point((int)tmp(0, 0), (int)tmp(0, 1)), cv::Point((int)tmp_(0, 0), (int)tmp_(0, 1)), cv::Scalar(255), 1);
		}
		//cv::imwrite("./select_feature.jpg", image);
	}
	
	
	void gbdt::print_tree()
	{
		for(size_t i = 0; i < tree_leaf_regression_value.size(); i++)
		{
			cv::Mat_<float> tmp = tree_leaf_regression_value[i].rowRange(0, 5).t();
			std::cout << tmp.rowRange(0, 1) << std::endl;
		}
	}
	/*void gbdt::load_model(std::ifstream &fin)
	{
		int split_node_num;
		fin >> split_node_num;
		int landmark_num;
		fin >> landmark_num;
		tree_node_split_feature.resize(split_node_num);
		tree_leaf_regression_value.resize(split_node_num + 1);
		for(int i = 0; i < split_node_num; i++)
			fin >> tree_node_split_feature[i].x >> tree_node_split_feature[i].y >> tree_node_split_feature[i].threshold;
		for(int i = 0; i < split_node_num + 1; i++){
			tree_leaf_regression_value[i] = (cv::Mat_<float>(landmark_num, 2)).zeros(landmark_num, 2);
			for(int j = 0; j < landmark_num; j++)
				fin >> tree_leaf_regression_value[i](j, 0) >> tree_leaf_regression_value[i](j, 1);
		}
	}
	
	void gbdt::save_model(std::ifstream &fout)
	{
		int split_node_num = tree_node_split_feature.size();
		fout >> split_node_num;
		int landmark_num = tree_leaf_regression_value[0].rows;
		fout >> landmark_num;
		//tree_node_split_feature.resize(split_node_num);
		//tree_leaf_regression_value.resize(split_node_num + 1);
		for(int i = 0; i < split_node_num; i++)
			fout << tree_node_split_feature[i].x << tree_node_split_feature[i].y << tree_node_split_feature[i].threshold;
		for(int i = 0; i < split_node_num + 1; i++){
			tree_leaf_regression_value[i] = (cv::Mat_<float>(landmark_num, 2)).zeros(landmark_num, 2);
			for(int j = 0; j < landmark_num; j++)
				fout << tree_leaf_regression_value[i](j, 0) << tree_leaf_regression_value[i](j, 1);
		}
	}*/
}