//============================================================================
// Name        : KinectCalib.cpp
// Author      : Robert Walter - github.com/robbeofficial
// Version     : 0.something
// Description : interactive calibration tool for kinect
//============================================================================

#include <iostream>
#include <sstream>
#include <fstream>
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
	CalibrationStereo,
	CalibrationCheck
};

/*//////////////////////////////////////////////////////////////////////
// constants
//////////////////////////////////////////////////////////////////////*/

const Size patternSize(8, 6);
const float patternTileWidth = 0.0289f; // in meters
const float patternTileHeight = 0.0289f;

const Size frameSize(640, 480); // TODO make use of it!
const Size subPixWindowSize(11, 11); 
const Size subPixZeroZone(-1, -1);
const TermCriteria subPixTerm(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.1);

const char *frameNamePrimary = "Primary Frame";
const char *frameNameSecondary = "Secondary Frame";

const Scalar textColorRed(0,0,255);
const Scalar textColorBlue(255,0,0);
const Scalar textColorWhite(255,255,255);
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
Mat R,T,E,F,rvec;

// image buffers
Mat irMat16(480, 640, CV_16UC1);
Mat irMat8(480, 640, CV_8UC1);
Mat irMat8Rgb(480, 640, CV_8UC3);
Mat irMat8RgbUndist(480, 640, CV_8UC1);

Mat rgbMat(480, 640, CV_8UC3); // note: this is actually BGR
Mat rgbMatGray(480, 640, CV_8UC1);
Mat rgbMatUndist(480, 640, CV_8UC3);

Mat depthMat16(480, 640, CV_16UC1);
Mat depthMat8(480, 640, CV_8UC1);

Mat overlayMat(480, 640, CV_8UC3);
Mat overlayCalibMat(480, 640, CV_8UC3);

/*//////////////////////////////////////////////////////////////////////
// functions
//////////////////////////////////////////////////////////////////////*/

void acquireIr(float alpha) {
	Mat mat(frameSize, CV_16UC1, (unsigned char*) ir.GetIRMap());
	mat.copyTo(irMat16);
	irMat16.convertTo(irMat8, CV_8UC1, alpha);
	cvtColor(irMat8, irMat8Rgb, CV_GRAY2RGB);
}

void acquireDepth(float alpha) {
	Mat mat(frameSize, CV_16UC1, (unsigned char*) depth.GetDepthMap());
	mat.copyTo(depthMat16);
	depthMat16.convertTo(depthMat8, CV_8UC1, alpha);
}

void textLine(Mat &img, const string line, int lineNumber, Scalar color) {
	putText(img, line, Point(textIndent,lineNumber*textHeight), textFont, textScale, color);
}

void acquireRgb() {
	Mat rgbTemp(frameSize, CV_8UC3, (unsigned char*) image.GetImageMap());
	cvtColor(rgbTemp, rgbMat, CV_RGB2BGR);
	cvtColor(rgbTemp, rgbMatGray, CV_RGB2GRAY);
}

void dump(Mat &m) {
	assert(m.type() == CV_64F);
	for (int j=0; j<m.size().height; j++) {
		for (int k=0; k<m.size().width; k++) {
			cout << (double) m.at<double>(j,k) << " ";
		}
		cout << endl;
	}
	cout <<  endl;
}

void dump(vector<Mat> &v) {
	for (int i=0; i<v.size(); i++) {
		assert(v[i].type()  == CV_64F);
		for (int j=0; j<v[i].size().height; j++) {
			for (int k=0; k<v[i].size().width; k++) {
				cout << (double) v[i].at<double>(j,k) << " ";
			}
			cout << endl;
		}
		cout <<  endl;
	}
}

void overlay(Mat &rgb, Mat &gray, Mat &overlay) {
	unsigned char *rgbData = (unsigned char *) rgb.data;
	unsigned char *grayData = (unsigned char *) gray.data;
	unsigned char *overlayData = (unsigned char *) overlay.data;

	int n = frameSize.width * frameSize.height;
	for (int i=0; i<n; i++) {
		if (grayData[i] == 0) {
			//overlayData[3*i+0] = grayData[i] == 0 ? 255 : 0;
			overlayData[3*i+1] = (rgbData[3*i+0] + rgbData[3*i+1] + rgbData[3*i+2]) / 3;
			overlayData[3*i+2] = 0;
		} else {
			overlayData[3*i+1] = (rgbData[3*i+0] + rgbData[3*i+1] + rgbData[3*i+2]) / 3;
			overlayData[3*i+2] = 255 - grayData[i];
		}
	}
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

void reconstruct(Mat &depthMat, Mat &cameraMatrix, Mat &distCoeffsMat, vector<Point3f> &points) {
	int width = depthMat.size().width;
	int height = depthMat.size().height;

	unsigned short *depth = (unsigned short *) depthMat.data;
	double *dist = (double *) distCoeffsMat.data;

	// undistort points (only once)
	static vector<Point2f> imgCoordsDist(width*height);
	static vector<Point2f> imgCoordsUndist(width*height);
	static bool firstCall = true;

	if (firstCall) {
		for (int u=0; u<width; u++) {
			for (int v=0; v<height; v++) {
				int i = u + v*width;
				imgCoordsDist[i] = Point2f(u,v);
			}
		}
		undistortPoints(Mat(imgCoordsDist), imgCoordsUndist, cameraMatrix, distCoeffsMat);
		firstCall = false;
	}
	//dump(imgCoordsUndist, 25);

	// reconstruct 3d coordinates
	for (int u=0; u<width; u++) {
		for (int v=0; v<height; v++) {
			int i = u + v*width;
			float xh = imgCoordsUndist[i].x;
			float yh = imgCoordsUndist[i].y;

			float Z = depth[i] / 1000.0f; // TODO pythagoras

			points[i] = Point3f(xh*Z, yh*Z, Z);
		}
	}
}

void mapDepthToRgb(Mat &src, Mat &dst, Mat rvec, Mat T, Mat irCameraMatrix, Mat irDistCoeffs, Mat rgbCameraMatrix, Mat rgbDistCoeffs) {
	// TODO
	static int n = frameSize.width*frameSize.height;
	static int width = frameSize.width;
	static vector<Point3f> reconstructedPoints(n);
	static vector<Point2f> projectedPoints(n);

	// reconstruct 3d points
	reconstruct(src, irCameraMatrix, irDistCoeffs, reconstructedPoints);

	// project 3d points to rgb coordinates
	projectPoints(Mat(reconstructedPoints), -rvec, -T, rgbCameraMatrix, rgbDistCoeffs, projectedPoints);

	// create calibrated depth map
	unsigned short *srcData = (unsigned short *) src.data;
	unsigned short *dstData = (unsigned short *) dst.data;
	for (int i=0; i<n; i++) {
		int u = (int) projectedPoints[i].x;
		int v = (int) projectedPoints[i].y;
		int j = u + v*width;
		if (j >= 0 && j < 640*480 && srcData[i] > 0) {
			//if (srcData[i] > dstData[j]) {
			if (dstData[j] == 0) {
				dstData[j] = srcData[i];
			} else {
				if (dstData[j] > srcData[i]) {
					dstData[j] = srcData[i];
				}
			}
				
			//}
		}
	}
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
	if (!irCameraMatrix.empty()) fs << "irCameraMatrix" << irCameraMatrix;
	if (!irDistCoeffs.empty()) fs << "irDistCoeffs" << irDistCoeffs;
	if (!rgbCameraMatrix.empty()) fs << "rgbCameraMatrix" << rgbCameraMatrix;
	if (!rgbDistCoeffs.empty()) fs << "rgbDistCoeffs" << rgbDistCoeffs;
	if (!R.empty()) fs << "R" << R;
	if (!T.empty()) fs << "T" << T;
	if (!E.empty()) fs << "E" << E;
	if (!F.empty()) fs << "F" << F;
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

	rvec.create(1,3,CV_64F);
	Rodrigues(R,rvec);

	cout << "irCameraMatrix" << endl; dump(irCameraMatrix);
	cout << "rgbCameraMatrix" << endl; dump(rgbCameraMatrix);
	cout << "T" << endl; dump(T);
	cout << "rvec" << endl; dump(rvec);

}

int main() {

	bool tracking = false;
	bool capture = false;

	// start in IR mode
	CalibrationMode mode = CalibrationIr;
	initCaptureMode(false, false, true);
	// DEBUG
//	CalibrationMode mode = CalibrationCheck;
//	initCaptureMode(true, true, false);
//	loadCalibration("calib1.home.xml"); // DEBUG

	int alpha = 5000;

	char key;
	
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
	
	while ( (key = (char) waitKey(1)) != 27 ) {
		//cout << "key : " << key << endl;
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
			case 'c':
				if (mode != CalibrationCheck) {
					mode = CalibrationCheck;
					context.Shutdown();
					initCaptureMode(true, true, false);
				}
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
			case CalibrationCheck:
				acquireDepth(255.0f / 4000.0f);
				acquireRgb();
				break;
		}
		
		// calibration
		switch (mode) {
			case CalibrationIr:			
				if (tracking) {
					cornersFound = findChessboardCorners(irMat8, patternSize, corners);
					if (cornersFound) {
						cornerSubPix(irMat8, corners, subPixWindowSize, subPixZeroZone, subPixTerm);
						drawChessboardCorners(irMat8Rgb, patternSize, Mat(corners), cornersFound);					
					}
					if (cornersFound && capture) {
						irObjPoints.push_back(objectPoints);
						irImgPoints.push_back(corners);
						error = calibrateCamera(
							irObjPoints, irImgPoints,
							frameSize,
							irCameraMatrix, irDistCoeffs,
							rvecs, tvecs
						);						
						cout << "irCameraMatrix" << endl; dump(irCameraMatrix);
						cout << "irDistCoeffs" << endl; dump(irDistCoeffs);
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
					if (cornersFound) {
						cornerSubPix(rgbMatGray, corners, subPixWindowSize, subPixZeroZone, subPixTerm);
						drawChessboardCorners(rgbMat, patternSize, Mat(corners), cornersFound);
					}
					if (cornersFound && capture) {
						rgbObjPoints.push_back(objectPoints);
						rgbImgPoints.push_back(corners);
						error = calibrateCamera(
							rgbObjPoints, rgbImgPoints,
							frameSize,
							rgbCameraMatrix, rgbDistCoeffs,
							rvecs, tvecs
						);
						cout << "rgbCameraMatrix" << endl; dump(rgbCameraMatrix);
						cout << "rgbDistCoeffs" << endl; dump(rgbDistCoeffs);						
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
					if (cornersFound) {
						cornerSubPix(rgbMatGray, corners, subPixWindowSize, subPixZeroZone, subPixTerm);
						drawChessboardCorners(rgbMat, patternSize, Mat(corners), cornersFound);					
					}
					if (cornersFound && capture) {
						// try to find points in rgb image as well
						context.Shutdown();
						initCaptureMode(false, false, true);
						context.WaitAndUpdateAll();
						acquireIr(alpha / 1000.0f);
						
						vector<Point2f> irCorners;
						cornersFound = findChessboardCorners(irMat8, patternSize, irCorners);
						if (cornersFound) {
							cornerSubPix(irMat8, corners, subPixWindowSize, subPixZeroZone, subPixTerm);
							drawChessboardCorners(irMat8Rgb, patternSize, Mat(irCorners), cornersFound);
						}
						
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
							Rodrigues(R,rvec);

							cout << "R" << endl; dump(R);
							cout << "T" << endl; dump(T);
							
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
			case CalibrationCheck:
			{
				// uncalibrated overlay
				overlay(rgbMat, depthMat8, overlayMat);
				textLine(overlayMat, "mode : Check", 1, textColorWhite);
				textLine(overlayMat, "uncalibrated overlay", 2, textColorWhite);

				// calibrated overlay
				Mat depthMat16Calib(480, 640, CV_16UC1);
				Mat depthMat8Calib(480, 640, CV_8UC1);
				depthMat16Calib.setTo(0);

				mapDepthToRgb(depthMat16, depthMat16Calib, rvec, T, irCameraMatrix, irDistCoeffs, rgbCameraMatrix, rgbDistCoeffs);
				depthMat16Calib.convertTo(depthMat8Calib, CV_8UC1, 255.0f / 4000.0f);
				overlay(rgbMat, depthMat8Calib, overlayCalibMat);
				textLine(overlayCalibMat, "calibrated overlay", 2, textColorWhite);

				imshow(frameNamePrimary, overlayMat);
				imshow(frameNameSecondary, overlayCalibMat);

				if (capture) { // capture json point clouds
					int n = frameSize.width*frameSize.height;
					int width = frameSize.width;

					vector<Point3f> reconstructedPoints(n);
					vector<Point2f> projectedPoints(n);

					// reconstruct 3d points
					reconstruct(depthMat16, irCameraMatrix, irDistCoeffs, reconstructedPoints);

					// project 3d points to rgb coordinates
					projectPoints(Mat(reconstructedPoints), -rvec, -T, rgbCameraMatrix, rgbDistCoeffs, projectedPoints);

		ofstream json;
  		json.open ("cloud.json");

		

					Mat thresh = depthMat16 < 1000 & depthMat16 > 0;
					unsigned char *threshData = (unsigned char *) thresh.data;
					unsigned char *rgbData = (unsigned char *) rgbMat.data;

					json << "points = [";
					for (int i=0; i<n; i++) {
						if (threshData[i]) {
							json << reconstructedPoints[i].x << ", " << reconstructedPoints[i].y << ", " << reconstructedPoints[i].z << ", " << endl;
						}
					}
					json << "];" << endl << endl;

					json << "colors = [";
					for (int i=0; i<n; i++) {
						if (threshData[i]) {
							int u = (int) projectedPoints[i].x;
							int v = (int) projectedPoints[i].y;
							int j = u + v*width;
							if (j >= 0 && j < n) {
								//json << reconstructedPoints[i].x << ", " << reconstructedPoints[i].y << ", " << reconstructedPoints[i].z << ", " << endl;
								json << (int)rgbData[3*j + 0] << ", " << (int)rgbData[3*j + 1] << ", " << (int)rgbData[3*j + 2] << ", " << endl;
							} else {
								json << "0, 0, 0," << endl;
							}
							
							//json << reconstructedPoints[i].x << ", " << reconstructedPoints[i].y << ", " << reconstructedPoints[i].z << ", " << endl;
						}
					}
					json << "];";

		json.close();

					//imshow("thresh" , thresh);
				}

				imshow("depthMat8", depthMat8);
				imshow("rgbMat", rgbMat);
	
				// DEBUG
//				Mat rgbTest(480, 640, CV_8UC1);
//				rgbTest.setTo(Scalar(0,0,0));
//				rgbMat.copyTo(rgbTest, depthMat16Calib < 1000 & depthMat16Calib > 0);
//				imshow("rgbTest", rgbTest);

			}
			break;

		}
	
		capture = false; // always capture only one view at a time
	}
	
}
