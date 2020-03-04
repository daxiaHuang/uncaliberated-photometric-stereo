#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "CF.h"
#include "uncalibrated_photometric_stereo.h"

template <typename T>
int sgn(T val) 
{
	return (T(0) < val) - (val < T(0));
}

static cv::Mat computeNormals(std::vector<cv::Mat> camImages, cv::Mat& Mask)
{
	int height = camImages[0].rows;
	int width = camImages[0].cols;
	int numImgs = camImages.size();
	/* populate A */
	cv::Mat A(height * width, numImgs, CV_32FC1, cv::Scalar::all(0));

	for (int k = 0; k < numImgs; k++) {
		int idx = 0;

		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				A.at<float>(idx++, k) = camImages[k].data[i * width + j] *
					sgn(Mask.at<uchar>(cv::Point(j, i)));
			}
		}
	}

	/* speeding up computation, SVD from A^TA instead of AA^T */
	cv::Mat U, S, Vt;
	cv::SVD::compute(A.t(), S, U, Vt, cv::SVD::MODIFY_A);
	cv::Mat EV = Vt.t();
	cv::Mat N(height, width, CV_8UC3, cv::Scalar::all(0));
	int idx = 0;

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (Mask.at<uchar>(cv::Point(j, i)) == 0) {
				N.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 255);
			}
			else {
				float rSxyz = 1.0f / sqrt(EV.at<float>(idx, 0) * EV.at<float>(idx, 0) +
					EV.at<float>(idx, 1) * EV.at<float>(idx, 1) +
					EV.at<float>(idx, 2) * EV.at<float>(idx, 2));
				/* V contains the eigenvectors of A^TA, which are as well the z,x,y
				 * components of the surface normals for each pixel	*/
				float sz = 128.0f +
					127.0f * sgn(EV.at<float>(idx, 0)) *
					fabs(EV.at<float>(idx, 0)) * rSxyz;
				float sx = 128.0f +
					127.0f * sgn(EV.at<float>(idx, 1)) *
					fabs(EV.at<float>(idx, 1)) * rSxyz;
				float sy = 128.0f +
					127.0f * sgn(EV.at<float>(idx, 2)) *
					fabs(EV.at<float>(idx, 2)) * rSxyz;

				N.at<cv::Vec3b>(i, j) = cv::Vec3b(sx, sy, sz);
			}

			idx += 1;
		}
	}

	return N;
}

int uncalibrated_photometric_stereo(const std::vector<cv::Mat>& images, cv::Mat& normalMap, cv::Mat& gradient)
{
	//判断输入参数
	if (images.size() < 3)//至少需要3张图像
	{
		return 0;
	}
	for (int i = 0; i < images.size(); ++i)
	{
		if (images[i].empty())
		{
			return 0;
		}
	}

	//计算法向图
	cv::Mat mask = cv::Mat(images[0].size(), CV_8UC1, cv::Scalar(255));
	normalMap = computeNormals(images, mask);
	cvtColor(normalMap, normalMap, cv::COLOR_BGR2RGB);
	//计算梯度图
	cv::Mat gx, gy, filterImg;
	cvtColor(normalMap, filterImg, cv::COLOR_RGB2GRAY);
	GaussianBlur(filterImg, filterImg, cv::Size(5, 5), 1.0, 0, cv::BORDER_DEFAULT);
	spatialGradient(filterImg, gx, gy);
	convertScaleAbs(gx, gx);
	convertScaleAbs(gy, gy);
	addWeighted(gx, 0.5, gy, 0.5, 0, gradient);
	gradient = ~gradient;
}

int derivate_vector_field(cv::Mat& image, cv::Mat& dst, int method)
{
	if (image.empty() || method > 4)
	{
		return 0;
	}

	int ItNum = 10;
	int Type = method;
	float lambda = 2.5f;
	float DataFitOrder = 1.0f;
	double mytime;
	//filter solver for the variational models
	CF *DualMesh = new CF;
	DualMesh->set(image);
	DualMesh->Solver(Type, mytime, ItNum, lambda, DataFitOrder);
	dst = DualMesh->get();
	cv::normalize(dst, dst, 150, 0, cv::NORM_MINMAX);
	dst.convertTo(dst, CV_8U);
	delete DualMesh;

	return 1;
}