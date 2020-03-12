#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "uncalibrated_photometric_stereo.h"

template <typename T>
int sgn(T val) 
{
	return (T(0) < val) - (val < T(0));
}

void updateHeights(cv::Mat &Normals, cv::Mat &Z, int iterations) {
	for (int k = 0; k < iterations; k++) {
		for (int i = 1; i < Normals.rows - 1; i++) {
			for (int j = 1; j < Normals.cols - 1; j++) {
				float zU = Z.at<float>(cv::Point(j, i - 1));
				float zD = Z.at<float>(cv::Point(j, i + 1));
				float zL = Z.at<float>(cv::Point(j - 1, i));
				float zR = Z.at<float>(cv::Point(j + 1, i));
				float nxC = Normals.at<cv::Vec3b>(cv::Point(j, i))[0];
				float nyC = Normals.at<cv::Vec3b>(cv::Point(j, i))[1];
				float nxU = Normals.at<cv::Vec3b>(cv::Point(j, i - 1))[0];
				float nyU = Normals.at<cv::Vec3b>(cv::Point(j, i - 1))[1];
				float nxD = Normals.at<cv::Vec3b>(cv::Point(j, i + 1))[0];
				float nyD = Normals.at<cv::Vec3b>(cv::Point(j, i + 1))[1];
				float nxL = Normals.at<cv::Vec3b>(cv::Point(j - 1, i))[0];
				float nyL = Normals.at<cv::Vec3b>(cv::Point(j - 1, i))[1];
				float nxR = Normals.at<cv::Vec3b>(cv::Point(j + 1, i))[0];
				float nyR = Normals.at<cv::Vec3b>(cv::Point(j + 1, i))[1];
				int up = nxU == 0 && nyU == 0 ? 0 : 1;
				int down = nxD == 0 && nyD == 0 ? 0 : 1;
				int left = nxL == 0 && nyL == 0 ? 0 : 1;
				int right = nxR == 0 && nyR == 0 ? 0 : 1;

				if (up > 0 && down > 0 && left > 0 && right > 0) {
					Z.at<float>(cv::Point(j, i)) =
						1.0f / 4.0f * (zD + zU + zR + zL + nxU - nxC + nyL - nyC);
				}
			}
		}
	}
}

cv::Mat localHeightfield(cv::Mat Normals) {
	const int pyramidLevels = 2;
	const int iterations = 5;
	/* building image pyramid */
	std::vector<cv::Mat> pyrNormals;
	cv::Mat Normalmap = Normals.clone();
	pyrNormals.push_back(Normalmap);

	for (int i = 0; i < pyramidLevels; i++) {
		cv::pyrDown(Normalmap, Normalmap);
		pyrNormals.push_back(Normalmap.clone());
	}

	/* updating depth map along pyramid levels, starting with smallest level at
	 * top */
	cv::Mat Z(pyrNormals[pyramidLevels - 1].rows,
		pyrNormals[pyramidLevels - 1].cols, CV_32FC1, cv::Scalar::all(0));

	for (int i = pyramidLevels - 1; i > 0; i--) {
		updateHeights(pyrNormals[i], Z, iterations);
		cv::pyrUp(Z, Z);
	}

	/* linear transformation of matrix values from [min,max] -> [a,b] */
	double min, max;
	cv::minMaxIdx(Z, &min, &max);
	double a = 0.0, b = 150.0;

	for (int i = 0; i < Normals.rows; i++) {
		for (int j = 0; j < Normals.cols; j++) {
			Z.at<float>(cv::Point(j, i)) =
				(float)a +
				(b - a) * ((Z.at<float>(cv::Point(j, i)) - min) / (max - min));
		}
	}

	return Z;
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
	gradient = computeNormals(images, mask);
}

//If use these filters to solve a complex data fitting term, define the data fitting as the blackbox function
static float BlackBox(int row, int col, cv::Mat& U, cv::Mat & img_orig, float & d)
{
	//this is an example of adaptive norm
	float diff = fabs(U.at<float>(row, col) + d - img_orig.at<float>(row, col));
	float order = 2 - (fabs(U.at<float>(row + 1, col) - U.at<float>(row, col)) + fabs(U.at<float>(row, col + 1) - U.at<float>(row, col)));
	return pow(diff, order);
}

int derivate_vector_field(cv::Mat& image, cv::Mat& dst)
{
	if (image.empty())
	{
		return 0;
	}

	dst = localHeightfield(image);
	cv::normalize(dst, dst, 255, 0, cv::NORM_MINMAX);
	dst.convertTo(dst, CV_8U);
	
	return 1;
}
