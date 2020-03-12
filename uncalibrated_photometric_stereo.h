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

*  @brief	光度立体算法

*  @param[in]	image 输入图像集

*  @param[out]	normalMap 计算得到的法向图

*  @param[out]	gradient 计算得到的梯度图

*  @Return		int

*  @note		至少需要三张不同方向的光源拍摄的相同物体的图像，固定相机和物体，从多个不同方向打光拍摄

*  @see			参考Halcon uncalibrated_photometric_stereo算子

*  @author		HuangLi

*  @date		2020/03/03 15:21

*/
int uncalibrated_photometric_stereo(const std::vector<cv::Mat>& images, cv::Mat& normalMap, cv::Mat& gradient);

/**

*  @brief	缺陷图构建

*  @param[in]	image 输入图像

*  @param[out]	dst 输出图像

*  @Return	int

*  @note	matters needing attention

*  @see    other functions

*  @author	HuangLi

*  @date	2020/03/03 16:28

*/
int derivate_vector_field(cv::Mat& image, cv::Mat& dst);