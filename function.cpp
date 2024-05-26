//function.cpp
/*
    Keval Visaria and Chirag Dhoka Jain

    This file has all the functions that are called in main.cpp 
*/
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <vector>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <cmath>
#include "function.h" // This will be updated to reflect new function signatures.

using namespace cv;
using namespace std;

/*Baseline Matching */
// Extract features from central square
Mat baselineMatching(const Mat& img) {
    Rect centerSquare(img.cols / 2 - 3, img.rows / 2 - 3, 7, 7);
    Mat centralFeatures = img(centerSquare).clone().reshape(1, 1);
    return centralFeatures;
}


// Derive color histogram features
Mat colorHistogramMatching(const Mat& img) {
    Mat hsvConverted;
    cvtColor(img, hsvConverted, COLOR_BGR2HSV);

    vector<Mat> hsvChannels;
    split(hsvConverted, hsvChannels);
    int h_bins = 8, s_bins = 8;
    int histogramSizes[] = { h_bins, s_bins };
    float hueRanges[] = { 0, 180 }, saturationRanges[] = { 0, 256 };
    const float* rangeSettings[] = { hueRanges, saturationRanges };
    int channels[] = { 0, 1 };
    Mat histogram;
    calcHist(&hsvConverted, 1, channels, Mat(), histogram, 2, histogramSizes, rangeSettings, true, false);
    normalize(histogram, histogram, 0, 1, NORM_MINMAX, -1, Mat());

    return histogram.reshape(1, 1);
}

// Generate full-image color histogram
Mat fullImageHistogram(const Mat& img) {
    Mat hsvConverted;
    cvtColor(img, hsvConverted, COLOR_BGR2HSV);

    vector<Mat> hsvChannels;
    split(hsvConverted, hsvChannels);
    int histogramBins[] = { 16, 16 };
    float hueRanges[] = { 0, 180 }, saturationRanges[] = { 0, 256 };
    const float* channelRanges[] = { hueRanges, saturationRanges };
    int channels[] = { 0, 1 };
    Mat histogram;
    calcHist(&hsvConverted, 1, channels, Mat(), histogram, 2, histogramBins, channelRanges, true, false);
    normalize(histogram, histogram, 0, 1, NORM_MINMAX, -1, Mat());

    return histogram.reshape(1, 1);
}

// Extract features from central region's histogram
Mat centralHistogramFeatures(const Mat& img) {
    Rect centerRegion(img.cols / 2 - 3, img.rows / 2 - 3, 7, 7);
    Mat centralRegion = img(centerRegion);

    Mat hsvCentral;
    cvtColor(centralRegion, hsvCentral, COLOR_BGR2HSV);

    vector<Mat> hsvChannels;
    split(hsvCentral, hsvChannels);
    int histogramBins[] = { 16, 16 };
    float hueRanges[] = { 0, 180 }, saturationRanges[] = { 0, 256 };
    const float* channelRanges[] = { hueRanges, saturationRanges };
    int channels[] = { 0, 1 };
    Mat histogram;
    calcHist(&hsvCentral, 1, channels, Mat(), histogram, 2, histogramBins, channelRanges, true, false);
    normalize(histogram, histogram, 0, 1, NORM_MINMAX, -1, Mat());

    return histogram.reshape(1, 1);
}

// Extract texture descriptors using gradient
Mat textureMatching(const Mat& img) {
    Mat grayImg;
    cvtColor(img, grayImg, COLOR_BGR2GRAY);

    Mat gradientX, gradientY;
    Sobel(grayImg, gradientX, CV_32F, 1, 0);
    Sobel(grayImg, gradientY, CV_32F, 0, 1);

    Mat magnitude, angle;
    cartToPolar(gradientX, gradientY, magnitude, angle, true);

    int histogramSize = 16;
    float range[] = { 0, 256 };
    const float* histRange = { range };
    Mat histogram;
    calcHist(&magnitude, 1, 0, Mat(), histogram, 1, &histogramSize, &histRange, true, false);
    normalize(histogram, histogram, 0, 1, NORM_MINMAX, -1, Mat());

    return histogram.reshape(1, 1);
}

// Implement a custom metric for feature comparison
double computeCustomMetric(const Mat& hist1, const Mat& hist2) {
    double intersectionVal = compareHist(hist1, hist2, HISTCMP_INTERSECT);
    double customScore = (0.6 * (1 - intersectionVal)) + (0.4 * intersectionVal);
    return customScore;
}

// Compute Euclidean distance between feature vectors
float computeEuclideanDistance(const vector<float>& vec1, const vector<float>& vec2) {
    float sum = 0.0;
    for (size_t i = 0; i < vec1.size(); ++i) {
        sum += pow(vec1[i] - vec2[i], 2);
    }
    return sqrt(sum);
}

// Retrieve feature vectors from a file
unordered_map<string, vector<float>> retrieveFeatureVectors(const string& filePath) {
    ifstream file(filePath);
    string line;
    unordered_map<string, vector<float>> featuresMap;

    while (getline(file, line)) {
        stringstream ss(line);
        string filename;
        getline(ss, filename, ',');

        vector<float> features(512);
        for (int i = 0; i < 512; ++i) {
            ss >> features[i];
            if (ss.peek() == ',') ss.ignore();
        }

        featuresMap[filename] = features;
    }

    return featuresMap;
}

// Process image retrieval using DNN features
vector<pair<double, string>> DNNBasedRetrieval(const string& directory, const string& featuresFile, const string& queryImage) {
    auto featuresMap = retrieveFeatureVectors(featuresFile);

    string queryFilename = queryImage.substr(queryImage.find_last_of("/\\") + 1);
    if (featuresMap.find(queryFilename) == featuresMap.end()) {
        cerr << "Query image features not found." << endl;
        return {};
    }
    vector<float> queryFeatures = featuresMap[queryFilename];

    vector<pair<double, string>> retrievalResults;
    for (const auto& featurePair : featuresMap) {
        if (featurePair.first == queryFilename) continue;
        float dist = computeEuclideanDistance(queryFeatures, featurePair.second);
        retrievalResults.emplace_back(dist, featurePair.first);
    }

    sort(retrievalResults.begin(), retrievalResults.end(), [](const pair<double, string>& a, const pair<double, string>& b) {
        return a.first < b.first;
        });

    return retrievalResults;
}

// Visualize an image in a window
void displayImage(const string& pathToImage) {
    Mat img = imread(pathToImage);
    if (img.empty()) {
        cout << "Error loading image: " << pathToImage << endl;
        return;
    }

    namedWindow("Display Window", WINDOW_NORMAL);
    imshow("Display Window", img);
    waitKey(0);
}

// LawsTexture
Mat computeLawsTextureFeatures(const Mat& image) {
    // Convert image to grayscale
    Mat grayImage;
    cvtColor(image, grayImage, COLOR_BGR2GRAY);

    // Define Laws' masks
    float L5[] = { 1,  4,  6,  4,  1 };
    float E5[] = { -1, -2,  0,  2,  1 };
    float S5[] = { -1,  0,  2,  0, -1 };
    float W5[] = { -1,  2,  0, -2,  1 };
    float R5[] = { 1, -4,  6, -4,  1 };

    // Compute convolutions using Laws' masks
    Mat L5E5, E5L5, L5L5, E5S5, S5E5, E5E5, S5S5, L5S5, S5L5;

    // Convert Laws' masks to Mat objects
    Mat L5Mat(1, 5, CV_32F, L5);
    Mat E5Mat(1, 5, CV_32F, E5);
    Mat S5Mat(1, 5, CV_32F, S5);

    filter2D(grayImage, L5E5, -1, L5Mat * E5Mat.t());
    filter2D(grayImage, E5L5, -1, E5Mat * L5Mat.t());
    filter2D(grayImage, L5L5, -1, L5Mat * L5Mat.t());
    filter2D(grayImage, E5S5, -1, E5Mat * S5Mat.t());
    filter2D(grayImage, S5E5, -1, S5Mat * E5Mat.t());
    filter2D(grayImage, E5E5, -1, E5Mat * E5Mat.t());
    filter2D(grayImage, S5S5, -1, S5Mat * S5Mat.t());
    filter2D(grayImage, L5S5, -1, L5Mat * S5Mat.t());
    filter2D(grayImage, S5L5, -1, S5Mat * L5Mat.t());


    // Compute energy values for each texture filter
    double energyL5E5 = norm(L5E5, NORM_L2);
    double energyE5L5 = norm(E5L5, NORM_L2);
    double energyL5L5 = norm(L5L5, NORM_L2);
    double energyE5S5 = norm(E5S5, NORM_L2);
    double energyS5E5 = norm(S5E5, NORM_L2);
    double energyE5E5 = norm(E5E5, NORM_L2);
    double energyS5S5 = norm(S5S5, NORM_L2);
    double energyL5S5 = norm(L5S5, NORM_L2);
    double energyS5L5 = norm(S5L5, NORM_L2);

    // Construct feature vector
    Mat features = (Mat_<float>(1, 9) << energyL5E5, energyE5L5, energyL5L5, energyE5S5, energyS5E5, energyE5E5, energyS5S5, energyL5S5, energyS5L5);

    return features;
}


// Function to compute Fourier Transform-based texture features in a single function
Mat computeFourierTextureFeatures(const Mat& image) {
    // Check if the input image is empty
    if (image.empty()) {
        throw std::invalid_argument("Input image is empty");
    }

    // Convert image to grayscale
    Mat grayImage;
    cvtColor(image, grayImage, COLOR_BGR2GRAY);

    // Convert the grayscale image to float32
    Mat floatImage;
    grayImage.convertTo(floatImage, CV_32F);

    // Compute 2D Fourier Transform
    Mat complexImage;
    dft(floatImage, complexImage, DFT_COMPLEX_OUTPUT);

    // Split the complex image into real and imaginary parts
    vector<Mat> planes;
    split(complexImage, planes);

    // Compute magnitude spectrum
    Mat magnitudeSpectrum;
    magnitude(planes[0], planes[1], magnitudeSpectrum);

    // Resize the magnitude spectrum to a 16x16 image
    Mat resizedSpectrum;
    resize(magnitudeSpectrum, resizedSpectrum, Size(16, 16));

    // Normalize the resized magnitude spectrum
    Mat normalizedSpectrum;
    normalize(resizedSpectrum, normalizedSpectrum, 0, 1, NORM_MINMAX);

    // Return the spectrum reshaped into a 1D feature vector
    return normalizedSpectrum.reshape(1, 1);
}

//Gabor Filter
Mat computeGaborHistogramFeatures(const Mat& image) {
    // Convert image to grayscale
    Mat grayImage;
    cvtColor(image, grayImage, COLOR_BGR2GRAY);

    // Define Gabor filter parameters
    int kernelSize = 7; // Adjust the kernel size as needed
    double sigma = 2.0;
    double lambda = 10.0;
    double gamma = 0.5;
    double psi = 0.0;

    // Initialize Gabor filter bank
    vector<Mat> gaborResponses;

    // Preallocate memory for filter responses
    gaborResponses.reserve(8);

    // Generate Gabor filter responses for different orientations
    for (int i = 0; i < 8; ++i) {
        double theta = i * CV_PI / 8.0;
        Mat gaborKernel = getGaborKernel(Size(kernelSize, kernelSize), sigma, theta, lambda, gamma, psi, CV_32F);

        Mat response;
        filter2D(grayImage, response, CV_32F, gaborKernel);
        gaborResponses.push_back(std::move(response)); // Move instead of copy
    }

    // Concatenate Gabor filter responses into a feature vector
    Mat gaborFeatureVector;
    hconcat(gaborResponses.data(), gaborResponses.size(), gaborFeatureVector);

    // Calculate histogram for the Gabor filter responses
    int histSize = 16; // Adjust the number of bins as needed
    float range[] = { 0, 256 };
    const float* histRange = { range };
    Mat hist;
    calcHist(&gaborFeatureVector, 1, 0, Mat(), hist, 1, &histSize, &histRange, true, false);
    normalize(hist, hist, 0, 1, NORM_MINMAX, -1, Mat());

    return hist.reshape(1, 1);
}

// Reading feature Vector
unordered_map<string, vector<float>> readFeatureVectorsFromCSV(const string& csvPath) {
    ifstream file(csvPath);
    string line;
    unordered_map<string, vector<float>> featureVectors;

    while (getline(file, line)) {
        stringstream ss(line);
        string filename;
        getline(ss, filename, ','); // Assuming the first column is the filename

        vector<float> features(516); // Assuming each vector has 512 elements + 2 Sobel gradients
        for (int i = 0; i < 512; ++i) {
            ss >> features[i];
            if (ss.peek() == ',') ss.ignore();
        }

        featureVectors[filename] = features;
    }

    return featureVectors;
}

//Finding Cosine Distance
float cosDistance(const vector<float>& v1, const vector<float>& v2) {
    float dotProduct = 0.0;
    float normV1 = 0.0;
    float normV2 = 0.0;

    for (size_t i = 0; i < v1.size(); ++i) {
        dotProduct += v1[i] * v2[i];
        normV1 += v1[i] * v1[i];
        normV2 += v2[i] * v2[i];
    }

    normV1 = sqrt(normV1);
    normV2 = sqrt(normV2);

    if (normV1 == 0.0 || normV2 == 0.0) {
        return 0.0; // Prevent division by zero
    }
    else {
        return dotProduct / (normV1 * normV2);
    }
}

vector<float> sobelGradients(const Mat& image) {
    Mat gray;
    cvtColor(image, gray, COLOR_BGR2GRAY);

    Mat sobelX, sobelY;
    Sobel(gray, sobelX, CV_64F, 1, 0);
    Sobel(gray, sobelY, CV_64F, 0, 1);

    Scalar meanSobelX = mean(sobelX);
    Scalar meanSobelY = mean(sobelY);

    vector<float> gradients = { static_cast<float>(meanSobelX[0]), static_cast<float>(meanSobelY[0]) };

    return gradients;
}

//DNN image Retrieval
vector<pair<double, string>> dnn_image_retrieval(const string& directory, const string& csvPath, const string& targetImagePath) {
    auto featureVectors = readFeatureVectorsFromCSV(csvPath);

    size_t pos = targetImagePath.find_last_of("/\\");
    string targetFilename = targetImagePath.substr(pos + 1);

    const auto& targetFeaturesIt = featureVectors.find(targetFilename);
    if (targetFeaturesIt == featureVectors.end()) {
        cerr << "Features for target image not found in CSV." << endl;
        return {};
    }
    const vector<float>& targetFeatures = targetFeaturesIt->second;

    Mat targetImage = imread(targetImagePath);
    vector<float> targetGradients = sobelGradients(targetImage);

    vector<pair<double, string>> similarities;
    for (const auto& pair : featureVectors) {
        if (pair.first == targetFilename) continue;

        float similarity = cosDistance(targetFeatures, pair.second);

        Mat image = imread(directory + "/" + pair.first);
        vector<float> gradients = sobelGradients(image);

        // Incorporate Sobel gradients into similarity computation
        float sobelSimilarity = 1 - (abs(targetGradients[0] - gradients[0]) + abs(targetGradients[1] - gradients[1])) / 255.0;

        // Combine DNN cosine similarity and Sobel similarity
        float combinedSimilarity = 0.5 * similarity + 0.5 * sobelSimilarity;
        similarities.emplace_back(combinedSimilarity, pair.first);
    }

    // Use lambda that captures by reference for sorting, if needed
    sort(similarities.begin(), similarities.end(), [](const pair<double, string>& a, const pair<double, string>& b) {
        return a.first > b.first; // Sort in descending order
        });

    return similarities;
}

