#ifndef FUNCTION_H
#define FUNCTION_H

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <unordered_map>

using namespace cv;
using namespace std;

// Declarations of the refactored functions
Mat baselineMatching(const Mat& img);
cv::Mat colorHistogramMatching(const cv::Mat& img);
cv::Mat fullImageHistogram(const cv::Mat& img);
cv::Mat textureMatching(const cv::Mat& img);
double computeCustomMetric(const cv::Mat& hist1, const cv::Mat& hist2);
cv::Mat centralHistogramFeatures(const cv::Mat& img);
float computeEuclideanDistance(const std::vector<float>& vec1, const std::vector<float>& vec2);
std::unordered_map<std::string, std::vector<float>> retrieveFeatureVectors(const std::string& filePath);
std::vector<std::pair<double, std::string>> DNNBasedRetrieval(const std::string& directory, const std::string& featuresFile, const std::string& queryImage);
void displayImage(const std::string& pathToImage);

Mat computeLawsTextureFeatures(const Mat& image);
Mat computeFourierTextureFeatures(const Mat& image);
Mat computeGaborHistogramFeatures(const Mat& image);

unordered_map<string, vector<float>> readFeatureVectorsFromCSV(const string& csvPath);
float cosDistance(const vector<float>& v1, const vector<float>& v2);
vector<float> sobelGradients(const Mat& image);
vector<pair<double, string>> dnn_image_retrieval(const string& directory, const string& csvPath, const string& targetImagePath);


#endif // FUNCTION_H
