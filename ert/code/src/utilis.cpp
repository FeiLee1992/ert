#include "utilis.h"

namespace ert{
	cv::Mat_<float> normalization(train_sample &sample)
	{
		cv::Mat_<float> scale_rotate;
		cv::Mat_<float> transform;
		cv::Mat_<float> point_from(4, 2);
		cv::Mat_<float> point_to(4, 2);
		cv::Mat_<float> result(sample.target_pts.rows, sample.target_pts.cols);
		point_from(0, 0) = sample.rect.x; point_from(0, 1) = sample.rect.y;
		point_from(1, 0) = sample.rect.x + sample.rect.width; point_from(1, 1) = sample.rect.y;
		point_from(2, 0) = sample.rect.x; point_from(2, 1) = sample.rect.y + sample.rect.height;
		point_from(3, 0) = sample.rect.x + sample.rect.width; point_from(3, 1) = sample.rect.y + sample.rect.height;
		point_to(0, 0) = 0; point_to(0, 1) = 0;
		point_to(1, 0) = 1; point_to(1, 1) = 0;
		point_to(2, 0) = 0; point_to(2, 1) = 1;
		point_to(3, 0) = 1; point_to(3, 1) = 1;
		get_similarity_transform(point_from, point_to, scale_rotate, transform);
		for(int i = 0; i < sample.target_pts.rows; i++)
		{
			cv::Mat_<float> tmp(1, 2);
			tmp(0, 0) = sample.target_pts(i, 0);
			tmp(0, 1) = sample.target_pts(i, 1);
			tmp = (scale_rotate * tmp.t()).t() + transform;
			result(i, 0) = tmp(0, 0);
			result(i, 1) = tmp(0, 1);
		}
		return result;
	}

	void unnormalization(train_sample &sample, cv::Mat_<float> &scale_rotate, cv::Mat_<float> &transform)
	{
		//cv::Mat_<float> scale_rotate;
		//cv::Mat_<float> transform;
		//std::cout << "-----------------" << std::endl;
		cv::Mat_<float> point_from(4, 2);
		cv::Mat_<float> point_to(4, 2);
		//cv::Mat_<float> result(sample.target_pts.rows, sample.target_pts.cols);
		point_to(0, 0) = sample.rect.x; point_to(0, 1) = sample.rect.y;
		point_to(1, 0) = sample.rect.x + sample.rect.width; point_to(1, 1) = sample.rect.y;
		point_to(2, 0) = sample.rect.x; point_to(2, 1) = sample.rect.y + sample.rect.height;
		point_to(3, 0) = sample.rect.x + sample.rect.width; point_to(3, 1) = sample.rect.y + sample.rect.height;
		point_from(0, 0) = 0; point_from(0, 1) = 0;
		point_from(1, 0) = 1; point_from(1, 1) = 0;
		point_from(2, 0) = 0; point_from(2, 1) = 1;
		point_from(3, 0) = 1; point_from(3, 1) = 1;
		//std::cout << "*****************" << std::endl;
		get_similarity_transform(point_from, point_to, scale_rotate, transform);
		//std::cout << "_________________" << std::endl;
		/*for(int i = 0; i < sample.target_pts.rows; i++)
		{
			cv::Mat_<float> tmp(1, 2);
			tmp(0, 0) = sample.cur_pts(i, 0);
			tmp(0, 1) = sample.cur_pts(i, 1);
			tmp = scale_rotate * tmp + transform;
			result(i, 0) = tmp(0, 0);
			result(i, 1) = tmp(0, 1);
		}*/
		//return result;
		//return result;
	}

	void get_similarity_transform(const cv::Mat_<float> &point_from, cv::Mat_<float> &point_to, cv::Mat_<float> &scale_rotate, cv::Mat_<float> &transform)
	{
		cv::Mat_<float> mean_from, mean_to;
		mean_from = (cv::Mat_<float>(1, 2)).zeros(1, 2);
		mean_to = (cv::Mat_<float>(1, 2)).zeros(1, 2);
		for(int i = 0; i < point_from.rows; i++)
		{
			mean_from(0, 0) += point_from(i, 0);
			mean_from(0, 1) += point_from(i, 1);
			mean_to(0, 0) += point_to(i, 0);
			mean_to(0, 1) += point_to(i, 1);
		}
		//std::cout << mean_from << ", " << mean_to << std::endl;
		mean_from /= point_from.rows;
		mean_to /= point_from.rows;
		float sigma_from = 0;
		float sigma_to = 0;
		cv::Mat_<float> cov = (cv::Mat_<float>(2, 2)).zeros(2, 2);
		for(int i = 0; i < point_from.rows; i++)
		{
			sigma_from += std::pow(point_from(i, 0) - mean_from(0, 0), 2) + std::pow(point_from(i, 1) - mean_from(0, 1), 2);
			sigma_to += std::pow(point_to(i, 0) - mean_to(0, 0), 2) + std::pow(point_to(i, 1) - mean_to(0, 1), 2);
			cov(0, 0) += (point_from(i, 0) - mean_from(0, 0)) * (point_to(i, 0) - mean_to(0, 0));
			cov(1, 1) += (point_from(i, 1) - mean_from(0, 1)) * (point_to(i, 1) - mean_to(0, 1));
			cov(0, 1) += (point_from(i, 0) - mean_from(0, 0)) * (point_to(i, 1) - mean_to(0, 1));
			cov(1, 0) += (point_from(i, 1) - mean_from(0, 1)) * (point_to(i, 0) - mean_to(0, 0));
		}

		sigma_from /= point_from.rows;
		sigma_to /= point_from.rows;
		//std::cout << cov << std::endl;
		cov /= point_from.rows;
		cv::Mat_<float> u, d, v, s;
		cv::SVD::compute(cov, d, u, v);

		s = (cv::Mat_<float>(2, 2)).zeros(2, 2);
		s(0, 0) = 1;
		s(1, 1) = 1;
		cv::Mat_<float> eigen = (cv::Mat_<float>(2, 2)).zeros(2, 2);
		eigen(0, 0) = d.at<float>(0);
		eigen(1, 1) = d.at<float>(1);
		if (cv::determinant(cov) < 0)
		{
			if (eigen(1, 1) < eigen(0, 0))
				s(1,1) = -1;
			else
				s(0,0) = -1;
		}
		
		cv::Mat_<float> r = u * s * v;

		float c = 1; 
		if (sigma_from != 0)
			c = 1.0 / sigma_from * cv::trace(eigen * s).val[0];

		cv::Mat_<float> t = mean_to - (c * r * mean_from.t()).t();

		scale_rotate = c * r;
		transform = t;
		//std::cout << t << std::endl;
		//std::cout << cov << std::endl << eigen << std::endl << u << std::endl << v << std::endl << r << std::endl << scale_rotate << std::endl << transform << std::endl;
		//return point_transform_affine(c*r, t);
	}
	
	void swap(train_sample *lhs, train_sample *rhs)
	{
		cv::Rect tmp;
		tmp.x = rhs->rect.x; tmp.y = rhs->rect.y; tmp.width = rhs->rect.width; tmp.height = rhs->rect.height;
		rhs->rect.x = lhs->rect.x; rhs->rect.y = lhs->rect.y; rhs->rect.width = lhs->rect.width; rhs->rect.height = lhs->rect.height;
		lhs->rect.x = tmp.x; lhs->rect.y = tmp.y; lhs->rect.width = tmp.width; lhs->rect.height = tmp.height;
			
		//cv::Mat_<float> _tmp = (cv::Mat_<float>(lhs.target_pts.rows, lhs.target_pts.cols)).zeros(lhs.target_pts.rows, lhs.target_pts.cols);
		//cv::Mat_<float> _tmp = rhs->target_pts.rowRange(0, 5).t() - rhs->cur_pts.rowRange(0, 5).t();
		//std::cout << _tmp.rowRange(0, 1) << std::endl;
		//_tmp = lhs->target_pts.rowRange(0, 5).t() - lhs->cur_pts.rowRange(0, 5).t();
		//std::cout << _tmp.rowRange(0, 1) << std::endl;
		for(int i = 0; i < rhs->target_pts.rows; i++)
		{
			cv::Mat_<float> _tmp(1, 2);
			_tmp(0, 0) = rhs->target_pts(i, 0);
			_tmp(0, 1) = rhs->target_pts(i, 1);
			rhs->target_pts(i, 0) = lhs->target_pts(i, 0);
			rhs->target_pts(i, 1) = lhs->target_pts(i, 1);
			lhs->target_pts(i, 0) = _tmp(0, 0);
			lhs->target_pts(i, 1) = _tmp(0, 1);
		}

		for(int i = 0; i < rhs->target_pts.rows; i++)
		{
			cv::Mat_<float> _tmp(1, 2);
			_tmp(0, 0) = rhs->cur_pts(i, 0);
			_tmp(0, 1) = rhs->cur_pts(i, 1);
			rhs->cur_pts(i, 0) = lhs->cur_pts(i, 0);
			rhs->cur_pts(i, 1) = lhs->cur_pts(i, 1);
			lhs->cur_pts(i, 0) = _tmp(0, 0);
			lhs->cur_pts(i, 1) = _tmp(0, 1);
		}
		//_tmp = rhs->target_pts.rowRange(0, 5).t() - rhs->cur_pts.rowRange(0, 5).t();
		//std::cout << _tmp.rowRange(0, 1) << std::endl;
		//_tmp = lhs->target_pts.rowRange(0, 5).t() - lhs->cur_pts.rowRange(0, 5).t();
		//std::cout << _tmp.rowRange(0, 1) << std::endl;
		//std::cout << std::endl;
		
		int index = rhs->index;
		rhs->index = lhs->index;
		lhs->index = index;
			
		
		for(int i = 0; i < rhs->feature_values.rows; i++)
		{
			cv::Mat_<int> tmp_(1, 1);
			tmp_(0, 0) = rhs->feature_values(i, 0);
			rhs->feature_values(i, 0) = lhs->feature_values(i, 0);
			lhs->feature_values(i, 0) = tmp_(0, 0);
		}
	}
	void copy(train_sample &lhs, train_sample &rhs)
	{
		lhs.index = rhs.index;
		
		lhs.rect.x = rhs.rect.x; lhs.rect.y = rhs.rect.y; lhs.rect.width = rhs.rect.width; lhs.rect.height = rhs.rect.height;
	
		lhs.target_pts = (cv::Mat_<float>(rhs.target_pts.rows, rhs.target_pts.cols)).zeros(rhs.target_pts.rows, rhs.target_pts.cols);
		
		for(int i = 0; i < rhs.target_pts.rows; i++)
		{
			lhs.target_pts(i, 0) = rhs.target_pts(i, 0);
			lhs.target_pts(i, 1) = rhs.target_pts(i, 1);
		}
	}
}