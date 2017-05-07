#include "regression.h"

#include <stdio.h>
#include <unistd.h>  
#include <dirent.h>  
#include <fstream>
#include <ctime>

void getFiles(std::string filePath, std::vector<std::string> &files)
{
    DIR *dir;  
    struct dirent *ptr;  
   
    if ((dir = opendir(filePath.c_str())) == NULL)  
    {  
		perror("Open dir error...");  
        exit(1);  
    }  
   
    while ((ptr = readdir(dir)) != NULL)  
    {  
        if(strcmp(ptr->d_name, ".") == 0 || strcmp(ptr->d_name, "..") == 0)
            continue;  
        else if(ptr->d_type == 8)
            files.push_back(ptr->d_name);  
        else if(ptr->d_type == 10)      
            continue;  
        else if(ptr->d_type == 4) 
        {  
            //files.push_back(ptr->d_name);  
        }  
    }  
    closedir(dir);  
}

int extractIndex(std::string &fileName)
{
	int num = 0;
	char appendix[8];
	sscanf(fileName.c_str(), "image_0%d.%s", &num, appendix);
	return num;
}

  
void quickSort(std::vector<std::string> &file, int start, int end)
{
	int mid = extractIndex(file[(start + end) / 2]);
	//std::cout << file[(start + end) / 2] << ", " << start << ": " << end << ", " << file[start] << ", " << file[end] << std::endl;
	int i = start;
	int j = end;
	while(i <= j)
	{
		while(extractIndex(file[i]) < mid && i <= j)
			i++;
		while(extractIndex(file[j]) > mid && i <= j)
			j--;
		if(i <= j)
		{
			std::string tmp = file[i];
			//std::cout << tmp << std::endl;
			file[i] = file[j];
			file[j] = tmp;
			i++;
			j--;
		}
	}
	if(start < j)
		quickSort(file, start, j);
	if(i < end)
		quickSort(file, i, end);
			
}	

bool ShapeInRect(cv::Mat_<float>& shape, cv::Rect& ret){
	float sum_x = 0.0, sum_y = 0.0;
	float max_x = 0, min_x = 10000, max_y = 0, min_y = 10000;
	for (int i = 0; i < shape.rows; i++){
		if (shape(i, 0)>max_x) max_x = shape(i, 0);
		if (shape(i, 0)<min_x) min_x = shape(i, 0);
		if (shape(i, 1)>max_y) max_y = shape(i, 1);
		if (shape(i, 1)<min_y) min_y = shape(i, 1);

		sum_x += shape(i, 0);
		sum_y += shape(i, 1);
	}
	sum_x /= shape.rows;
	sum_y /= shape.rows;

	if ((max_x - min_x) > ret.width * 1.5) return false;
	if ((max_y - min_y) > ret.height * 1.5) return false;
	if (std::abs(sum_x - (ret.x + ret.width / 2.0)) > ret.width / 1.5) return false;
	if (std::abs(sum_y - (ret.y + ret.height / 2.0)) > ret.height / 1.5) return false;
	return true;
}

bool ShapeInRect(cv::Mat_<float>& shape, cv::Rect& ret, int landmark_num){
	float sum_x = 0.0, sum_y = 0.0;
	float max_x = 0, min_x = 10000, max_y = 0, min_y = 10000;
	for (int i = 0; i < shape.rows; i++){
		if (shape(i, 0)>max_x) max_x = shape(i, 0);
		if (shape(i, 0)<min_x) min_x = shape(i, 0);
		if (shape(i, 1)>max_y) max_y = shape(i, 1);
		if (shape(i, 1)<min_y) min_y = shape(i, 1);

		sum_x += shape(i, 0);
		sum_y += shape(i, 1);
	}
	sum_x /= shape.rows;
	sum_y /= shape.rows;
	if(landmark_num == 5)
	{
		if(min_x > ret.x && min_x < ret.x + ret.width &&
			min_y > ret.y && min_y < ret.y + ret.height)
			return true;
		else
			return false;
	}
	if ((max_x - min_x) > ret.width * 1.5) return false;
	if ((max_y - min_y) > ret.height * 1.5) return false;
	if (std::abs(sum_x - (ret.x + ret.width / 2.0)) > ret.width / 1.5) return false;
	if (std::abs(sum_y - (ret.y + ret.height / 2.0)) > ret.height / 1.5) return false;
	return true;
}

void LoadImages(std::vector<cv::Mat_<uchar> > &images, std::vector<cv::Rect> &bounding_rects, std::vector<cv::Mat_<float> > &ground_truth_shapes,
	std::string filename, std::string filePath)
{
	//bool init = true;
	std::string fn_haar = "../haarcascade_frontalface_alt2.xml";
    cv::CascadeClassifier haar_cascade;
    bool yes = haar_cascade.load(fn_haar);
    //std::cout << "face detector loaded : " << yes << std::endl;
    //std::cout << "loading images..." << std::endl;
	std::vector<std::string> png_files;
	std::vector<std::string> pts_files;
	//getFiles(filePath + "/png", png_files);
	//getFiles(filePath + "/pts", pts_files);
	//quickSort(png_files, 0, png_files.size() - 1);
	//quickSort(pts_files, 0, pts_files.size() - 1);
	//int cnt = png_files.size();
	//std::cout << "png total num: " << cnt << std::endl;
	//cnt = 100;
	std::ifstream fin;
	fin.open(filename.c_str(), std::fstream::in);
	while(true){
		std::string png_name, pts_name;
		fin >> png_name >> pts_name;
		//std::cout << png_name << std::endl;
		if(png_name.empty())
			break;
		png_files.push_back(png_name);
		pts_files.push_back(pts_name);
	}
	int cnt = png_files.size();
	//cnt = 100;
	std::cout << "find png cnt: " << cnt << std::endl;
	//std::string filePath("../dataset/helen/trainset/");
	int count = 0;
	//int landmarks = landmark_num;
	for(int i = 0; i < cnt; i++)
	{
		int landmarks;
		std::string file_name = filePath + png_files[i];
		cv::Mat_<uchar> image = cv::imread(file_name.c_str(), 0);
		
		file_name = filePath + pts_files[i];

		std::ifstream fin;
		std::string temp;
		fin.open(file_name.c_str(), std::fstream::in);
		getline(fin, temp);
		fin >> temp >> landmarks;
		
		//if(init)
		//{
		//	std::cout << "landmarks num: " << landmarks << " image width, height: (" << image.cols << ", " << image.rows << ")" << std::endl;
		//	init = false;
		//	memset(_mean_shape, 0, sizeof(float) * landmarks * 2);
		//}
		cv::Mat_<float> ground_truth_shape(landmarks, 2);
		getline(fin, temp); 
		getline(fin, temp); 
		for (int m = 0; m < landmarks; m++){
			fin >> ground_truth_shape(m, 0) >> ground_truth_shape(m, 1);
		}
		fin.close();
		
		if (image.cols > 2000){
            cv::resize(image, image, cv::Size(image.cols / 3, image.rows / 3), 0, 0, cv::INTER_LINEAR);
			ground_truth_shape / 3;
        }
        else if (image.cols > 1400 && image.cols <= 2000){
            cv::resize(image, image, cv::Size(image.cols / 3, image.rows / 3), 0, 0, cv::INTER_LINEAR);
			ground_truth_shape / 3;
        }
		std::vector<cv::Rect> faces;
        haar_cascade.detectMultiScale(image, faces, 1.1, 2, 0, cv::Size(30, 30));

        for (int m = 0; m < faces.size(); m++){
            cv::Rect faceRec = faces[m];
			//faceRec.x = faceRec.x - faceRec.width / 5;
			//faceRec.y = faceRec.y - faceRec.height / 5;
			//faceRec.width += faceRec.width / 5 * 2;
			//faceRec.height += faceRec.height / 5 * 2;
            if (ShapeInRect(ground_truth_shape, faceRec)){
				//struct ImageLabel imageLabel;
				//image.copyTo(imageLabel.img);
				images.push_back(image);
				//imageLabel.landmarkPos = new float[landmark_num * 2];
				//for(int k = 0; k < landmarks; k++)
				//{
				//	imageLabel.landmarkPos[k] = ground_truth_shape(k, 0);
				//	_mean_shape[k] += (ground_truth_shape(k, 0) - (faceRec.x + faceRec.width) / 2) / faceRec.width - 0.5;
				//	imageLabel.landmarkPos[k + landmarks] = ground_truth_shape(k, 1);
				//	_mean_shape[k + landmarks] += (ground_truth_shape(k, 1) - (faceRec.y + faceRec.height) / 2) / faceRec.height - 0.5;
				//}
				ground_truth_shapes.push_back(ground_truth_shape);
				//imageLabel.faceBox[0] = faceRec.x;
				//imageLabel.faceBox[1] = faceRec.y;
				//imageLabel.faceBox[2] = faceRec.width;
				//imageLabel.faceBox[3] = faceRec.height;
				bounding_rects.push_back(faceRec);
                //imageLabels.push_back(imageLabel);
                count++;
                if (count%100 == 0){
                    std::cout << count << " images loaded\n";
                }
                break;
            }
        }
	}
	std::cout << "load " << count << " images" << std::endl;
	//for(int i = 0; i < landmarks; i++)
	//{
	//	_mean_shape[i] /= imageLabels.size();
	//	_mean_shape[i + landmarks] /= imageLabels.size();
	//}
}


void LoadImages(std::vector<cv::Mat_<uchar> > &images, std::vector<cv::Rect> &bounding_rects, std::vector<cv::Mat_<float> > &ground_truth_shapes,
	std::string filePath)
{
	//bool init = true;
	std::cout << "load image in " << filePath << std::endl;
	std::string fn_haar = "../haarcascade_frontalface_alt2.xml";
    cv::CascadeClassifier haar_cascade;
    bool yes = haar_cascade.load(fn_haar);
    //std::cout << "face detector loaded : " << yes << std::endl;
    //std::cout << "loading images..." << std::endl;
	std::vector<std::string> png_files;
	std::vector<std::string> pts_files;
	getFiles(filePath + "/png", png_files);
	getFiles(filePath + "/pts", pts_files);
	quickSort(png_files, 0, png_files.size() - 1);
	quickSort(pts_files, 0, pts_files.size() - 1);
	int cnt = png_files.size();
	std::cout << "png total num: " << cnt << std::endl;
	cnt = 100;
	int static i = 0;
	if(i++ == 1)
		cnt = png_files.size();
	int count = 0;
	//int landmarks = landmark_num;
	for(int i = 0; i < cnt; i++)
	{
		int landmarks;
		std::string file_name = filePath + "/png/" + png_files[i];
		cv::Mat_<uchar> image = cv::imread(file_name.c_str(), 0);
		
		file_name = filePath + "/pts/" + pts_files[i];

		std::ifstream fin;
		std::string temp;
		fin.open(file_name.c_str(), std::fstream::in);
		getline(fin, temp);
		fin >> temp >> landmarks;
		
		//if(init)
		//{
		//	std::cout << "landmarks num: " << landmarks << " image width, height: (" << image.cols << ", " << image.rows << ")" << std::endl;
		//	init = false;
		//	memset(_mean_shape, 0, sizeof(float) * landmarks * 2);
		//}
		cv::Mat_<float> ground_truth_shape(landmarks, 2);
		getline(fin, temp); 
		getline(fin, temp); 
		for (int m = 0; m < landmarks; m++){
			fin >> ground_truth_shape(m, 0) >> ground_truth_shape(m, 1);
		}
		fin.close();
		
		if (image.cols > 2000){
            cv::resize(image, image, cv::Size(image.cols / 4, image.rows / 4), 0, 0, cv::INTER_LINEAR);
			ground_truth_shape / 4;
        }
        else if (image.cols > 1400 && image.cols <= 2000){
            cv::resize(image, image, cv::Size(image.cols / 3, image.rows / 3), 0, 0, cv::INTER_LINEAR);
			ground_truth_shape / 3;
        }
		std::vector<cv::Rect> faces;
        haar_cascade.detectMultiScale(image, faces, 1.1, 2, 0, cv::Size(30, 30));

        for (int m = 0; m < faces.size(); m++){
            cv::Rect faceRec = faces[m];
			//faceRec.x = faceRec.x - faceRec.width / 5;
			//faceRec.y = faceRec.y - faceRec.height / 5;
			//faceRec.width += faceRec.width / 5 * 2;
			//faceRec.height += faceRec.height / 5 * 2;
            if (ShapeInRect(ground_truth_shape, faceRec)){
				//struct ImageLabel imageLabel;
				//image.copyTo(imageLabel.img);
				images.push_back(image);
				//imageLabel.landmarkPos = new float[landmark_num * 2];
				//for(int k = 0; k < landmarks; k++)
				//{
				//	imageLabel.landmarkPos[k] = ground_truth_shape(k, 0);
				//	_mean_shape[k] += (ground_truth_shape(k, 0) - (faceRec.x + faceRec.width) / 2) / faceRec.width - 0.5;
				//	imageLabel.landmarkPos[k + landmarks] = ground_truth_shape(k, 1);
				//	_mean_shape[k + landmarks] += (ground_truth_shape(k, 1) - (faceRec.y + faceRec.height) / 2) / faceRec.height - 0.5;
				//}
				ground_truth_shapes.push_back(ground_truth_shape);
				//imageLabel.faceBox[0] = faceRec.x;
				//imageLabel.faceBox[1] = faceRec.y;
				//imageLabel.faceBox[2] = faceRec.width;
				//imageLabel.faceBox[3] = faceRec.height;
				bounding_rects.push_back(faceRec);
						
                //imageLabels.push_back(imageLabel);
                count++;
                if (count%100 == 0){
                    std::cout << count << " images loaded\n";
                }
                break;
            }
        }
	}
	//std::cout << "load " << count << " images" << std::endl;
	//for(int i = 0; i < landmarks; i++)
	//{
	//	_mean_shape[i] /= imageLabels.size();
	//	_mean_shape[i + landmarks] /= imageLabels.size();
	//}
}

void LoadImages(std::vector<cv::Mat_<uchar> > &train_images, 
	std::vector<cv::Rect> &train_bounding_rects, 
	std::vector<cv::Mat_<float> > &train_ground_truth_shapes,
	std::vector<cv::Mat_<uchar> > &val_images, 
	std::vector<cv::Rect> &val_bounding_rects, 
	std::vector<cv::Mat_<float> > &val_ground_truth_shapes,
	int landmarks)
{

	{
		int width, height;
		FILE *file = fopen("./train_data.dat", "rb");
		int cnt;
		fread(&cnt, sizeof(int), 1, file);
		int index = 0;
		
		while(index < 165000)
		{
			fread(&height, sizeof(int), 1, file);
			fread(&width, sizeof(int), 1, file);
			cv::Mat_<uchar> img(height, width);
			fread(img.data, width * height * sizeof(uchar), 1, file);
			fread(&height, sizeof(int), 1, file);
			fread(&width, sizeof(int), 1, file);
			cv::Mat_<float> shape(height, width);
			fread(shape.data, width * height * sizeof(float), 1, file);
			cv::Rect rect;
			fread(&rect, sizeof(cv::Rect), 1, file);
			train_images.push_back(img);
			train_bounding_rects.push_back(rect);
			train_ground_truth_shapes.push_back(shape);
			index++;
		}
		//cv::imwrite("./lifei.jpg", train_images[0]);
		index = 0;
		while(index < 2000)
		{
			fread(&height, sizeof(int), 1, file);
			fread(&width, sizeof(int), 1, file);
			cv::Mat_<uchar> img(height, width);
			fread(img.data, width * height * sizeof(uchar), 1, file);
			fread(&height, sizeof(int), 1, file);
			fread(&width, sizeof(int), 1, file);
			cv::Mat_<float> shape(height, width);
			fread(shape.data, width * height * sizeof(float), 1, file);
			cv::Rect rect;
			fread(&rect, sizeof(cv::Rect), 1, file);
			val_images.push_back(img);
			val_bounding_rects.push_back(rect);
			val_ground_truth_shapes.push_back(shape);
			index++;
		}
		fclose(file);
	}
	return;

	std::cout << "load images and landmark in five point mode" << std::endl;
	std::string fn_haar = "./../haarcascade_frontalface_alt2.xml";
	cv::CascadeClassifier haar_cascade;
	bool yes = haar_cascade.load(fn_haar);
	std::cout << "face detector loaded : " << yes << std::endl;
	std::cout << "loading images..." << std::endl;
	int count = 0;
	std::vector<std::string> image_path_prefix, landmarks_list, tag_image_path_prefix;
	landmarks_list.push_back("./../../alignment/dataset/list_landmarks_celeba.txt");
	tag_image_path_prefix.push_back("./../../alignment/dataset/list_eval_partition.txt");
	image_path_prefix.push_back("./../../alignment/dataset/celeba");

	for (int i = 0; i < image_path_prefix.size(); i++) {
		std::ifstream fin;
		std::ifstream fmark;
		fin.open((landmarks_list[i]).c_str(), std::ifstream::in);
		fmark.open(tag_image_path_prefix[i].c_str(), std::ifstream::in);
		std::string path_prefix = image_path_prefix[i];
		std::string image_file_name, image_pts_name;
		std::cout << "loading images in folder: " << path_prefix << std::endl;
		int sample_cnt = 0;
		fin >> sample_cnt;

		std::string temp;
		getline(fin, temp);
		getline(fin, temp);
		int index = 0, img_status = 0;
		//std::cout << temp << std::endl;
		while(train_images.size() < 182000 && val_images.size() < 2000)
		{
			cv::Mat_<float> ground_truth_shape(landmarks, 2);
			fin >> image_file_name;
			fmark >> temp >> img_status;
			if(img_status == 2)
				break;
			for (int i = 0; i < landmarks; i++)
				fin >> ground_truth_shape[i][0] >> ground_truth_shape[i][1];

			std::string image_path;
			if (path_prefix[path_prefix.size() - 1] == '/')
				image_path = path_prefix + image_file_name;
			else
				image_path = path_prefix + "/" + image_file_name;
			//std::cout << image_path << std::endl;
			cv::Mat_<uchar> image = cv::imread(image_path.c_str(), 0);
			if (image.cols > 1500){
				cv::resize(image, image, cv::Size(image.cols / 4, image.rows / 4), 0, 0, cv::INTER_LINEAR);
				ground_truth_shape /= 4.0;
			}
			else if (image.cols > 800 && image.cols <= 1500){
				cv::resize(image, image, cv::Size(image.cols / 3, image.rows / 3), 0, 0, cv::INTER_LINEAR);
				ground_truth_shape /= 3.0;
			}

			std::vector<cv::Rect> faces;
			haar_cascade.detectMultiScale(image, faces, 1.1, 2, 0, cv::Size(15, 15));

			for (int i = 0; i < faces.size(); i++){
				cv::Rect faceRec = faces[i];
				if (ShapeInRect(ground_truth_shape, faceRec, landmarks)){
					if(train_images.size() < 180000)
					{
						train_images.push_back(image);
						train_bounding_rects.push_back(faceRec);
						train_ground_truth_shapes.push_back(ground_truth_shape);
					}
					else{
						val_images.push_back(image);
						val_bounding_rects.push_back(faceRec);
						val_ground_truth_shapes.push_back(ground_truth_shape);
					}
					count++;
					
					if (count % 100 == 0){
						std::cout << count << " images loaded\n";
					}
					break;
				}
			}
			index++;
		}
		fin.close();
		fmark.close();
	}
	//std::ifstream in;
	//save images(train, test)
	{
		int width, height;
		FILE *file = fopen("./train_data.dat", "wb");
		int cnt = train_images.size();
		fwrite(&cnt, sizeof(int), 1, file);
		for(int i = 0; i < train_images.size(); i++)
		{
			height = train_images[i].rows;
			width = train_images[i].cols;
		
			fwrite(&height, sizeof(int), 1, file);
			fwrite(&width, sizeof(int), 1, file);
			fwrite(train_images[i].data, width * height * sizeof(uchar), 1, file);
		
			height = train_ground_truth_shapes[i].rows;
			width = train_ground_truth_shapes[i].cols;
		
			fwrite(&height, sizeof(int), 1, file);
			fwrite(&width, sizeof(int), 1, file);
			fwrite(train_ground_truth_shapes[i].data, width * height * sizeof(float), 1, file);
		
			//std::cout << ground_truth_shapes[i] << std::endl;
			fwrite(&train_bounding_rects[i], sizeof(cv::Rect), 1, file);
		}
		cnt = val_images.size();
		fwrite(&cnt , sizeof(cnt), 1, file);
		for(int i = 0; i < val_images.size(); i++)
		{
			height = val_images[i].rows;
			width = val_images[i].cols;
			
			fwrite(&height, sizeof(int), 1, file);
			fwrite(&width, sizeof(int), 1, file);
			fwrite(val_images[i].data, width * height * sizeof(uchar), 1, file);
		
			height = val_ground_truth_shapes[i].rows;
			width = val_ground_truth_shapes[i].cols;
			
			fwrite(&height, sizeof(int), 1, file);
			fwrite(&width, sizeof(int), 1, file);
			fwrite(val_ground_truth_shapes[i].data, height * width * sizeof(float), 1, file);
			
			fwrite(&val_bounding_rects[i], sizeof(cv::Rect), 1, file);
		}
		fclose(file);
	}
	std::cout << "get " << train_images.size() << " faces in train group" << std::endl;
}

void augment_train_set(std::vector<cv::Mat_<uchar> > &train_images, std::vector<cv::Rect> &train_bounding_box, std::vector<cv::Mat_<float> > &train_ground_truth_shapes)
{
	std::cout << "augment train data, before augmenting: " << train_images.size() << std::endl;
	size_t cnt = train_images.size();
	for(int l = 0; l < 2; l++)
		for(size_t i = 0; i < cnt; i++)
		{
			cv::Point2f center = cv::Point2f(train_images[i].cols / 2, train_images[i].rows / 2);
			double angle = (5 + std::rand() % 20) * ((l % 2 == 0) ? 1 : -1);
			double scale = 1;
			cv::Mat rotateMat;   
			rotateMat = cv::getRotationMatrix2D(center, angle, scale);  
			//std::cout << rotateMat << std::endl;
			cv::Mat_<float> tmp(train_ground_truth_shapes[i].rows, train_ground_truth_shapes[i].cols);
			
			for(int k = 0; k < train_ground_truth_shapes[i].rows; k++)
			{
				tmp(k, 0) = train_ground_truth_shapes[i](k, 0) * rotateMat.at<double>(0, 0) + train_ground_truth_shapes[i](k, 1) * rotateMat.at<double>(0, 1) + rotateMat.at<double>(0, 2);
				tmp(k, 1) = train_ground_truth_shapes[i](k, 0) * rotateMat.at<double>(1, 0) + train_ground_truth_shapes[i](k, 1) * rotateMat.at<double>(1, 1) + rotateMat.at<double>(1, 2);		
			}
			//std::cout << tmp.t() << std::endl;
			train_ground_truth_shapes.push_back(tmp);
			
			cv::Rect tmp_rect = train_bounding_box[i];
			//tmp_rect.x = train_bounding_box[i].x * rotateMat.at<double>(0, 0) + train_bounding_box[i].y * rotateMat.at<double>(0, 1) + rotateMat.at<double>(0, 2);
			//tmp_rect.y = train_bounding_box[i].x * rotateMat.at<double>(1, 0) + train_bounding_box[i].y * rotateMat.at<double>(1, 1) + rotateMat.at<double>(1, 2);

			train_bounding_box.push_back(tmp_rect);
			
			cv::Mat rotateImg;  
			cv::warpAffine(train_images[i], rotateImg, rotateMat, train_images[i].size());  
			train_images.push_back(rotateImg);
			//char buf[32];
			//memset(buf, 0, sizeof(buf));
			//sprintf(buf, "./augment/%d.jpg", i);
			//for(int k = 0; k < tmp.rows; k++)
			//	cv::circle(rotateImg, cv::Point(tmp(k, 0), tmp(k, 1)), 3, cv::Scalar(255, 255, 255));
			//cv::rectangle(rotateImg, tmp_rect, cv::Scalar(255, 255, 255), 2, 2);
			//imwrite(buf, rotateImg);
		}
	std::cout << "origin size: " << cnt << " cur size: " << train_images.size() << std::endl;
}


int main(int argv, char **argc)
{
	std::srand(std::time(0));
	
	std::vector<cv::Mat_<uchar> > train_images;
	std::vector<cv::Rect> train_bounding_box;
	std::vector<cv::Mat_<float> > train_ground_truth_shapes;
	
	std::vector<cv::Mat_<uchar> > val_images;
	std::vector<cv::Rect> val_bounding_rects; 
	std::vector<cv::Mat_<float> > val_ground_truth_shapes;
	LoadImages(train_images, train_bounding_box, train_ground_truth_shapes, val_images, val_bounding_rects, val_ground_truth_shapes, atoi(argc[1]));
	
	//LoadImages(train_images, train_bounding_box, train_ground_truth_shapes, "./../../alignment/dataset/lfpw/trainset");
	//augment_train_set(train_images, train_bounding_box, train_ground_truth_shapes);
	//LoadImages(train_images, train_bounding_box, train_ground_truth_shapes, "./../dataset/helen_train_images_list.txt", "./../dataset/helen/trainset/");
	std::cout << "load images: " << train_images.size() << std::endl;
	ert::regression regressors(700, 14, 400, 1, 0.1);
	regressors.train(train_images, train_bounding_box, train_ground_truth_shapes, 5, 20, 0.18, 0.1);
	
	train_images.clear();
	train_bounding_box.clear();
	train_ground_truth_shapes.clear();
	//LoadImages(val_images, val_bounding_rects, val_ground_truth_shapes, "./../../alignment/dataset/lfpw/testset");
	//LoadImages(val_images, val_bounding_rects, val_ground_truth_shapes, "./../dataset/helen_test_images_list.txt", "./../dataset/helen/testset/");
	float total_error = 0;
	for(size_t i = 0; i < val_images.size(); i++)
	{
		//std::cout << bounding_box[i].x << ", " << bounding_box[i].y << ", " << bounding_box[i].width << ", " << bounding_box[i].height << std::endl;
		cv::Mat_<float> predict_shapes;
		clock_t time_begin = clock();
		//std::cout << predict_shapes << std::endl;
		regressors.predict(val_images[i], predict_shapes, val_bounding_rects[i]);
		clock_t time_end = clock();
		//std::cout << "time: " << (time_end - time_begin) / (float)CLOCKS_PER_SEC << std::endl;
		//regressors.validate_similarity_transform(bounding_box[i], ground_truth_shapes[i]);
		//std::cout << bounding_box[i] << std::endl;
		cv::Mat_<float> diff = predict_shapes - val_ground_truth_shapes[i];
		float error = 0;
		float inter_ocular = 0;
		for(int j = 0; j < predict_shapes.rows; j++){
            float x = predict_shapes(j, 0);
            float y = predict_shapes(j, 1);
            cv::circle(val_images[i], cv::Point(x, y), 3, cv::Scalar(255, 255, 255), -1);
			error += std::sqrt(std::pow(diff(j, 0), 2) + std::pow(diff(j, 1), 2));
			x = val_ground_truth_shapes[i](j, 0);
			y = val_ground_truth_shapes[i](j, 1);
			cv::circle(val_images[std::pow(i, 1)], cv::Point(x, y), 1, cv::Scalar(0, 0, 0), -1);
        }
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
		float lx = 0, ly = 0, rx = 0, ry = 0;
		for(int k = ll_bound; k <= lr_bound; k++){
			lx += val_ground_truth_shapes[i](k, 0);
			ly += val_ground_truth_shapes[i](k, 1);
		}
		for(int k = rl_bound; k <= rr_bound; k++)
		{
			rx += val_ground_truth_shapes[i](k, 0);
			ry += val_ground_truth_shapes[i](k, 1);
		}
		lx /= (1 + lr_bound - ll_bound);
		ly /= (1 + lr_bound - ll_bound);
		rx /= (1 + lr_bound - ll_bound);
		ry /= (1 + lr_bound - ll_bound);
		inter_ocular = std::sqrt(std::pow(lx - rx, 2) + std::pow(ly - ry, 2));
		//std::cout << "err: " << error / inter_ocular / predict_shapes.rows << std::endl;
		//if(i > (images.size() / 2))
		total_error += error / inter_ocular / predict_shapes.rows;
		//std::cout << "error: " << error / inter_ocular / predict_shapes.rows << std::endl;
		//std::cout << predict_shapes << std::endl;
		cv::rectangle(val_images[i], val_bounding_rects[i], cv::Scalar(255, 255, 255), 1, 1, 0);
		char tmp_buf[128];
		memset(tmp_buf, 0, sizeof(tmp_buf));
		sprintf(tmp_buf, "./result/%d.jpg", (int)i);
		cv::imwrite(tmp_buf, val_images[i]);
	}
	std::cout << "test error: " << total_error / (val_images.size()) << std::endl;
}