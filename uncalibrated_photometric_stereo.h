/**

* @file    uncalibrated_photometric_stereo.h

* @brief    This is a brief description.

* @author    HuangLi

* @date   2020/03/03 14:51

* @note   matters needing attention

* @version

*/

#include <opencv2/core.hpp>

/**

*  @brief	��������㷨

*  @param[in]	image ����ͼ��

*  @param[out]	normalMap ����õ��ķ���ͼ

*  @param[out]	gradient ����õ����ݶ�ͼ

*  @Return		int

*  @note		������Ҫ���Ų�ͬ����Ĺ�Դ�������ͬ�����ͼ�񣬹̶���������壬�Ӷ����ͬ����������

*  @see			�ο�Halcon uncalibrated_photometric_stereo����

*  @author		HuangLi

*  @date		2020/03/03 15:21

*/
int uncalibrated_photometric_stereo(const std::vector<cv::Mat>& images, cv::Mat& normalMap, cv::Mat& gradient);

/**

*  @brief	ȱ��ͼ����

*  @param[in]	image ����ͼ��

*  @param[out]	dst ���ͼ��

*  @Return	int

*  @note	matters needing attention

*  @see    other functions

*  @author	HuangLi

*  @date	2020/03/03 16:28

*/
int derivate_vector_field(cv::Mat& image, cv::Mat& dst);