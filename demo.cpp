#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "uncalibrated_photometric_stereo.h"
#include "type.h"

/**
* 功能：光度立体算法用于缺陷检测，至少需要三张
* step0：读取图像
* step1：计算法向图
* step2：计算缺陷图
* step3：对缺陷图处理
**/

int main(int argc, char *argv[])
{
	/********************step0：读取图像***************************/
	int numPics = 4;
	std::vector<cv::Mat> camImages;
	for (int i = 0; i < numPics; i++) 
	{
		std::stringstream s;
		s << "./images/image" << i << ".png";
		camImages.push_back(cv::imread(s.str(), cv::IMREAD_GRAYSCALE));
	}
  
	/********************step1：计算法向图***************************/
	cv::Mat normalMap, gradient;
	uncalibrated_photometric_stereo(camImages, normalMap, gradient);

	/********************step2：计算缺陷图***************************/
	cv::Mat defectImg;
	derivate_vector_field(gradient, defectImg, Curvature_Mean);


	return 0;
}
