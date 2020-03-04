#ifndef _TYPE_H
#define _TYPE_H


//! mode of the filter algorithm
enum FilterType {
    /*Image Median Filter*/
    Filter_Median  = 0,
    /*Image Mean Filter*/
    Filter_Mean    = 1,
    /*Image Gauss	Filter*/
    Filter_Gauss   = 2,
    /*Image FFT low pass Filter*/
    Filter_LowPass = 3,
};


//! mode of the Curvature algorithm
enum CurvatureType {
	Curvature_Variation		= 0,
	Curvature_Mean			= 1,
	Curvature_Difference	= 2,
	Curvature_Gaussian		= 3,
	Curvature_Bernstein		= 4,
};



#endif