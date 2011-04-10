//============================================================================
// Name        : KinectCalib.cpp
// Author      : Robert Walter - github.com/robbeofficial
// Version     : 0.something
// Description : interactive calibration tool for kinect
//============================================================================
/* estimates camera matrix, lens distortion coefficients of IR and RGB camera
 * as well as extrinsic stereo calibration including rotation and translation 
 * between sensor and essential- plus fundamental matrix.
 *
 * requires openCV 2.2!
 *
 * uses a planar b/w chessboard pattern as calibration object. you can simply
 * print one your and and glue it onto a paperboard to improve the results.
 * just be sure to adjust the following constants in the code:
 * patternSize, patternTileWidth, patternTileWidth
 *
 * depth image acquistion is not used for the calibration! only raw IR and
 * RGB images are used. as the structured light pattern of the IR projector
 * may trouble the pattern detection it is highly recommended to simply
 * cover the projector (leftmost lens) of the kinect. in this case you have
 * to provide other IR light sources. sunlight is just perfect but also some
 * lamps (usually not energy saving lamps) emmit a decent amount of IR light. 
 * you have to experiment a little bit. 
 *
 * for pattern detection and tracking, the 16 bit IR image needs to be 
 * quantized down to 8 bits. use the alpha slider to optimize the brightness
 * of the target image!
 *
 * key mapping:
 *
 *		'i'				switch to IR mode (calibrate IR camera)
 *
 * 		'r'				switch to RGB mode (calibrate RGB camera)
 *
 *		'q'				switch to stereo mode (stereo calibration)
 *							requires IR and RGB camera to be calibrated well!
 *		
 *		't'				toggles calibration pattern tracking mode
 *
 *		SPACE			captures a view and calibrates based on all already
 *							captured views
 *		
 *		's'				saves calibration data (calib.xml)
 *
 *		'l'				loads calibration data (calib.xml)
 *
 * TODOs:
 *		
 *		- figure out why results are still very poor (at least for me)
 *		- better feedback of stereo calibration quality (depth+IR overlay)
 *
 */

#include <iostream>
#include <sstream>
using namespace std;

#include <opencv/cv.h>
#include <opencv/highgui.h>
using namespace cv;

#include <XnCppWrapper.h>
using namespace xn;

// parallel sensor usage:
// RGB+DEPTH => OK
// RGB+IR => FAIL
// DEPTH+IR => OK

// sensor switching:
// IR->RGB => slow
// RGB->IR => fast

/*//////////////////////////////////////////////////////////////////////
// types
//////////////////////////////////////////////////////////////////////*/

enum CalibrationMode {
	CalibrationIr,
	CalibrationRgb,
	CalibrationStereo
};

/*//////////////////////////////////////////////////////////////////////
// constants
//////////////////////////////////////////////////////////////////////*/

const Size patternSize(8, 6);
const float patternTileWidth = 0.0289f; // in meters
const float patternTileHeight = 0.0289f;

const Size frameSize(640, 480); // TODO make use of it!

const char *frameNamePrimary = "Primary Frame";
const char *frameNameSecondary = "Secondary Frame";

const Scalar textColorRed(0,0,255);
const Scalar textColorBlue(255,0,0);
const int textFont = FONT_HERSHEY_SIMPLEX;
const float textScale = 1.0f;
const int textIndent = 10;
const int textHeight = 30;

/*//////////////////////////////////////////////////////////////////////
// globals
//////////////////////////////////////////////////////////////////////*/

// openNI context and generator nodes
Context context;
ImageGenerator image;
DepthGenerator depth;
IRGenerator ir;

// IR camera point correspondences (multiple views)
vector<vector<Point3f> > irObjPoints;
vector<vector<Point2f> > irImgPoints;

// RGB camera point correspondences (multiple views)
vector<vector<Point3f> > rgbObjPoints;
vector<vector<Point2f> > rgbImgPoints;

// IR+RGB stere correspondences
vector<vector<Point3f> > stereoObjPoints;
vector<vector<Point2f> > stereoIrImgPoints;
vector<vector<Point2f> > stereoRgbImgPoints;

// IR camera calibration data
Mat irCameraMatrix;
Mat irDistCoeffs;

// RGB camera calibration data
Mat rgbCameraMatrix;
Mat rgbDistCoeffs;

// stereo calibration
Mat R,T,E,F;

// image buffers
Mat irMat16(480, 640, CV_16UC1);
Mat irMat8(480, 640, CV_8UC1);
Mat irMat8Rgb(480, 640, CV_8UC3);
Mat irMat8RgbUndist(480, 640, CV_8UC1);

Mat rgbMat(480, 640, CV_8UC3); // note: this is actually BGR
Mat rgbMatUndist(480, 640, CV_8UC3);

Mat depthMat16(480, 640, CV_16UC1);
Mat depthMat8(480, 640, CV_8UC1);	

/*//////////////////////////////////////////////////////////////////////
// functions
//////////////////////////////////////////////////////////////////////*/

void acquireIr(float alpha) {
	Mat mat(frameSize, CV_16UC1, (unsigned char*) ir.GetIRMap());
	mat.copyTo(irMat16);
	irMat16.convertTo(irMat8, CV_8UC1, alpha);
	cvtColor(irMat8, irMat8Rgb, CV_GRAY2RGB);
}

void textLine(Mat &img, const string line, int lineNumber, Scalar color) {
	putText(img, line, Point(textIndent,lineNumber*textHeight), textFont, textScale, color);
}

void acquireRgb() {
	Mat rgbTemp(frameSize, CV_8UC3, (unsigned char*) image.GetImageMap());
	cvtColor(rgbTemp, rgbMat, CV_RGB2BGR);
}

void initCaptureMode(bool initImage, bool initDepth, bool initIR) {
	XnStatus nRetVal = XN_STATUS_OK;

	// Initialize context object
	nRetVal = context.Init();
	cout << "init : " << xnGetStatusString(nRetVal) << endl;

	// default output mode
	XnMapOutputMode outputMode;
	outputMode.nXRes = 640;
	outputMode.nYRes = 480;
	outputMode.nFPS = 30;

	// Create an ImageGenerator node
	if (initImage) {	
		nRetVal = image.Create(context);
		cout << "image.Create : " << xnGetStatusString(nRetVal) << endl;
		nRetVal = image.SetMapOutputMode(outputMode);
		cout << "image.SetMapOutputMode : " << xnGetStatusString(nRetVal) << endl;
	}

	// Create a DepthGenerator node	
	if (initDepth) {
		nRetVal = depth.Create(context);
		cout << "depth.Create : " << xnGetStatusString(nRetVal) << endl;
		nRetVal = depth.SetMapOutputMode(outputMode);
		cout << "depth.SetMapOutputMode : " << xnGetStatusString(nRetVal) << endl;
	}

	// Create an IRGenerator node
	if (initIR) {
		nRetVal = ir.Create(context);
		cout << "ir.Create : " << xnGetStatusString(nRetVal) << endl;
		nRetVal = ir.SetMapOutputMode(outputMode);
		cout << "ir.SetMapOutputMode : " << xnGetStatusString(nRetVal) << endl;
	}

	// Make it start generating data
	nRetVal = context.StartGeneratingAll();
	cout << "context.StartGeneratingAll : " << xnGetStatusString(nRetVal) << endl;
}

void projectDepthToRgb(const Mat &src, Mat &dst, Mat R, Mat T, Mat irCameraMatrix, Mat irDistCoeffs, Mat rgbCameraMatrix, Mat rgbDistCoeffs) {
	// TODO
}

void genPatternObjectPoints(Size patternSize, float dx, float dy, vector<Point3f> &objectPoints) {
	int w = patternSize.width, h = patternSize.height;
	for (int i=0; i<h; i++) {
		for (int j=0; j<w; j++) {
			objectPoints.push_back( Point3f(i*dx,j*dy,0) );
		}
	}
}

void saveCalibration(const char *fname) {
	FileStorage fs(fname, FileStorage::WRITE);
	fs << "irCameraMatrix" << irCameraMatrix;
	fs << "irDistCoeffs" << irDistCoeffs;
	fs << "rgbCameraMatrix" << rgbCameraMatrix;
	fs << "rgbDistCoeffs" << rgbDistCoeffs;
	fs << "R" << R;
	fs << "T" << T;
	fs << "E" << E;
	fs << "F" << F;
}

void loadCalibration(const char *fname) {
	FileStorage fs(fname, FileStorage::READ);
	fs["irCameraMatrix"] >> irCameraMatrix;
	fs["irDistCoeffs"] >> irDistCoeffs;
	fs["rgbCameraMatrix"] >> rgbCameraMatrix;
	fs["rgbDistCoeffs"] >> rgbDistCoeffs;
	fs["R"] >> R;
	fs["T"] >> T;
	fs["E"] >> E;
	fs["F"] >> F;
}

int main() {

	bool tracking = false;
	bool capture = false;

	// start in IR mode
	CalibrationMode mode = CalibrationIr;
	initCaptureMode(false, false, true);

	int alpha = 5000;

	int key;
	
	stringstream sstream;

	// create windows
	namedWindow(frameNamePrimary);
	createTrackbar("IR alpha", frameNamePrimary, &alpha, 10000);
	
	vector<Point3f> objectPoints;
	vector<Point2f> corners;
	bool cornersFound;
	double error = 0;
	
	vector<Mat> rvecs; // unneeded
	vector<Mat> tvecs;
	
	genPatternObjectPoints(patternSize, patternTileWidth, patternTileHeight, objectPoints);
	
	while ( (key = waitKey(1)) != 27 ) {
		// handle keys
		switch (key) {
			case 't':
				tracking = !tracking;
				break;
			case 'i':
				if (mode != CalibrationIr) {
					mode = CalibrationIr;
					context.Shutdown();
					initCaptureMode(false, false, true);
				}
				break;
			case 'r':
				if (mode != CalibrationRgb) {
					mode = CalibrationRgb;
					context.Shutdown();
					initCaptureMode(true, false, false);
				}
				break;
			case 's':
				saveCalibration("calib.xml");
				break;
			case 'l':
				loadCalibration("calib.xml");
				break;
			case 'q':
				if (mode != CalibrationStereo) {
					mode = CalibrationStereo;
					context.Shutdown();
					initCaptureMode(true, false, false);
				}
				break;
			case 32:
				capture = true;
				break;
		}
		
		// acquire sensor data
		context.WaitAndUpdateAll();
		switch (mode) {
			case CalibrationIr:			
				acquireIr(alpha / 1000.0f);
				break;
			case CalibrationStereo:
			case CalibrationRgb:
				acquireRgb();
				break;
		}
		
		// calibration
		switch (mode) {
			case CalibrationIr:			
				if (tracking) {
					cornersFound = findChessboardCorners(irMat8, patternSize, corners);
					drawChessboardCorners(irMat8Rgb, patternSize, corners, cornersFound);					
					if (cornersFound && capture) {
						irObjPoints.push_back(objectPoints);
						irImgPoints.push_back(corners);
						error = calibrateCamera(
							irObjPoints, irImgPoints,
							frameSize,
							irCameraMatrix, irDistCoeffs,
							rvecs, tvecs
						);
						if (!irCameraMatrix.empty()) {
							undistort(irMat8Rgb, irMat8RgbUndist, irCameraMatrix, irDistCoeffs);
							textLine(irMat8RgbUndist, "undistorted using calibration", 1, textColorBlue);
							if (irObjPoints.size() > 0) {
								sstream.str(""); sstream << "number of views : " << irImgPoints.size();
								textLine(irMat8RgbUndist, sstream.str(), 2, textColorBlue);

								sstream.str(""); sstream << "reprojection error : " << error;
								textLine(irMat8RgbUndist, sstream.str(), 3, textColorBlue);
							}					
							imshow(frameNameSecondary, irMat8RgbUndist);							
						}
					}
					textLine(irMat8Rgb, "tracking", 3, textColorRed);
					if (cornersFound) {
						textLine(irMat8Rgb, "press 'SPACE' to capture view", 2, textColorBlue);
					} else {
						textLine(irMat8Rgb, "show me the calibration pattern", 2, textColorBlue);
					}
				} else {
					textLine(irMat8Rgb, "press 't' to toggle pattern tracking", 2, textColorBlue);
				}				

				textLine(irMat8Rgb, "mode : IR", 1, textColorBlue);
				imshow(frameNamePrimary, irMat8Rgb);	
				
				break;
				
			case CalibrationRgb:
				if (tracking) {
					cornersFound = findChessboardCorners(rgbMat, patternSize, corners);
					drawChessboardCorners(rgbMat, patternSize, corners, cornersFound);					
					if (cornersFound && capture) {
						rgbObjPoints.push_back(objectPoints);
						rgbImgPoints.push_back(corners);
						error = calibrateCamera(
							rgbObjPoints, rgbImgPoints,
							frameSize,
							rgbCameraMatrix, rgbDistCoeffs,
							rvecs, tvecs
						);
						if (!rgbCameraMatrix.empty()) {
							undistort(rgbMat, rgbMatUndist, rgbCameraMatrix, rgbDistCoeffs);

							textLine(rgbMatUndist, "undistorted using calibration", 1, textColorBlue);
							if (rgbObjPoints.size() > 0) {
								sstream.str(""); sstream << "number of views : " << rgbImgPoints.size();
								textLine(rgbMatUndist, sstream.str(), 2, textColorBlue);

								sstream.str(""); sstream << "reprojection error : " << error;
								textLine(rgbMatUndist, sstream.str(), 3, textColorBlue);
							}									
					
							imshow(frameNameSecondary, rgbMatUndist);
						}
					}
					textLine(rgbMat, "tracking", 3, textColorRed);
					if (cornersFound) {
						textLine(rgbMat, "press 'SPACE' to capture view", 2, textColorBlue);
					} else {
						textLine(rgbMat, "show me the calibration pattern", 2, textColorBlue);
					}					
				} else {
					textLine(rgbMat, "press 't' to toggle pattern tracking", 2, textColorBlue);
				}
				
				textLine(rgbMat, "mode : RGB", 1, textColorBlue);
				imshow(frameNamePrimary, rgbMat);	
				
				break;
			case CalibrationStereo:
				if (tracking) {
					cornersFound = findChessboardCorners(rgbMat, patternSize, corners);
					drawChessboardCorners(rgbMat, patternSize, corners, cornersFound);					
					if (cornersFound && capture) {
						// try to find points in rgb image as well
						context.Shutdown();
						initCaptureMode(false, false, true);
						context.WaitAndUpdateAll();
						acquireIr(alpha / 1000.0f);
						
						vector<Point2f> irCorners;
						cornersFound = findChessboardCorners(irMat8, patternSize, irCorners);
						drawChessboardCorners(irMat8Rgb, patternSize, irCorners, cornersFound);
						
						if (cornersFound) { // we have found a set of correspondences!
							stereoObjPoints.push_back(objectPoints);
							stereoIrImgPoints.push_back(irCorners);
							stereoRgbImgPoints.push_back(corners);

							error = stereoCalibrate(
								stereoObjPoints,
								stereoRgbImgPoints, stereoIrImgPoints,
								rgbCameraMatrix, rgbDistCoeffs,
								irCameraMatrix, irDistCoeffs,								
								frameSize, R, T,
								E, F
							);
							
							if (stereoObjPoints.size() > 0) {
								sstream.str(""); sstream << "number of views : " << stereoObjPoints.size();
								textLine(irMat8Rgb, sstream.str(), 2, textColorBlue);

								sstream.str(""); sstream << "reprojection error : " << error;
								textLine(irMat8Rgb, sstream.str(), 3, textColorBlue);
							}
						} else {
							textLine(irMat8Rgb, "pattern not found, try again!", 4, textColorRed);
						}
						textLine(irMat8Rgb, "view from IR cam", 1, textColorBlue);
						
						imshow(frameNamePrimary, irMat8Rgb); 
						waitKey(1); // force repaint because ...
						
						// ... this is going to take a lot of time
						context.Shutdown();
						initCaptureMode(true, false, false);
					}
					textLine(rgbMat, "tracking", 3, textColorRed);
					if (cornersFound) {
						textLine(rgbMat, "press 'SPACE' to capture view", 2, textColorBlue);
					} else {
						textLine(rgbMat, "show me the calibration pattern", 2, textColorBlue);
					}					
				} else {
					textLine(rgbMat, "press 't' to toggle pattern tracking", 2, textColorBlue);
				}

				textLine(rgbMat, "mode : Stereo", 1, textColorBlue);
				imshow(frameNameSecondary, rgbMat);
			
				break;			

		}
	
		capture = false; // always capture only one view at a time
	}
	
}
