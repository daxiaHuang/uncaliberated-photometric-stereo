#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "uncalibrated_photometric_stereo.h"

/**
* ���ܣ���������㷨����ȱ�ݼ�⣬������Ҫ����
* step0����ȡͼ��
* step1�����㷨��ͼ
* step2������ȱ��ͼ
* step3����ȱ��ͼ����
**/

int main(int argc, char *argv[])
{
	/********************step0����ȡͼ��***************************/
	int numPics = 4;
	std::vector<cv::Mat> camImages;
	for (int i = 0; i < numPics; i++) 
	{
		std::stringstream s;
		s << "./images/image" << i << ".png";
		cv::Mat src = cv::imread(s.str(), cv::IMREAD_GRAYSCALE);
		camImages.push_back(src.clone());
	}
  
	/********************step1�����㷨��ͼ***************************/
	cv::Mat normalMap, gradient;
	uncalibrated_photometric_stereo(camImages, normalMap, gradient);

	/********************step2������ȱ��ͼ***************************/
	cv::Mat defectImg;
	derivate_vector_field(gradient, defectImg);

	return 0;
}
