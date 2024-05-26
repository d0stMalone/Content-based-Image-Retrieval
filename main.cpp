//main.cpp
// File: main.cpp
// Author: Keval Visaria and Chirag Dhoka Jain
// Date: Feb 10, 2024
// Compute feature functions and gives the necessary outputs

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <vector>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <cmath>
#include "function.h"

using namespace cv;
using namespace std;

const string DIRECTORY = "C:/Users/visar/Desktop/OneDrive - Northeastern University/PRCV/olympus";
const string TARGET_IMAGE = "C:/Users/visar/Desktop/OneDrive - Northeastern University/PRCV/olympus/pic.0088.jpg";
const string CSV_PATH = "C:/Users/visar/Desktop/OneDrive - Northeastern University/PRCV/ResNet18_olym.csv";

int main() {

    Mat targetImage = imread(TARGET_IMAGE);
    if (targetImage.empty()) {
        std::cout << "Error: Unable to read target image " << targetImage << endl;
        return -1;
    }
    Mat targetBaselineFeatures = baselineMatching(targetImage);
    Mat targetHistogramFeatures = colorHistogramMatching(targetImage);
    Mat targetWholeImageHist = fullImageHistogram(targetImage);
    Mat targetCenterRegionHist = centralHistogramFeatures(targetImage);
    Mat targetTextureFeatures = textureMatching(targetImage);
    Mat targetLawsTextureFeatures = computeLawsTextureFeatures(targetImage);
    Mat targetFourierTextureFeatures = computeFourierTextureFeatures(targetImage);
    Mat targetGaborHistogramFeatures = computeGaborHistogramFeatures(targetImage);

    vector<string> filenames;
    glob(DIRECTORY + "/*.jpg", filenames);

    while (true) {

        imshow("Target Image", targetImage);

        // Feature type menu
        std::cout << "Select FEATURE_TYPE:" << std::endl;
        std::cout << "1. BASELINE MATCHING" << std::endl;
        std::cout << "2. HISTOGRAM MATCHING" << std::endl;
        std::cout << "3. MULTI-HISTOGRAM MATCHING" << std::endl;
        std::cout << "4. TEXTURE AND COLOR MATCHING" << std::endl;
        std::cout << "5. DEEP NETWORK EMBEDDINGS" << std::endl;
        std::cout << "6. CUSTOM DESIGN" << std::endl;
        std::cout << "7. FACE DETECTION" << std::endl;
        std::cout << "Enter choice (1-7): ";
        int choice;
        std::cin >> choice;
        
        // Reset input state in case of a failed extraction
        cin.clear();
        cin.ignore(numeric_limits<streamsize>::max(), '\n');

        // Handling for invalid choice not covered explicitly
        if (choice < 1 || choice > 7) {
            std::cout << "Invalid choice, please try again." << std::endl;
            continue; // Skip the rest of the loop and show the menu again
        }

        std::string featureType;

        // Ask for NO._OF_IMAGES
        std::cout << "Enter NO._OF_IMAGES: ";
        int numberOfImages;
        std::cin >> numberOfImages;
        if (std::cin.fail()) { // if user input is not an integer
            std::cerr << "Error: NO._OF_IMAGES must be an integer." << std::endl;
            return 1; // Exit with error code
        }

        // Output the selected options (for demonstration purposes)
        std::cout << "Selected FEATURE_TYPE: " << featureType << std::endl;
        std::cout << "NO._OF_IMAGES: " << numberOfImages << std::endl;

        //Baseline Matching
        if (choice == 1) {

            featureType = "BASELINE MATCHING";

            std::cout << "Baseline Matching in Progres..." << std::endl;

            // Calculate distances for baseline features
            vector<pair<double, string>> distances;

            for (const auto& filename : filenames) {

                // Exclude the target image
                if (filename == TARGET_IMAGE) continue;

                // Read image
                Mat image = imread(filename);
                if (image.empty()) {
                    std::cout << "Error: Unable to read image " << filename << std::endl;
                    continue;
                }

                // Compute features
                Mat imageBaselineFeatures = baselineMatching(image);

                // Compute distance
                double distance = norm(targetBaselineFeatures, imageBaselineFeatures, NORM_L2);
                distances.push_back(make_pair(distance, filename));
            }

            // Sort distances
            sort(distances.begin(), distances.end());

            // Display top N baseline matches
            std::cout << "Top " << numberOfImages << " baseline matches for the target image " << TARGET_IMAGE << ":" << endl;
            for (int i = 0; i < numberOfImages && i < distances.size(); ++i) {
                std::cout << distances[i].second << " (Distance: " << distances[i].first << ")" << endl;
            }

            // Display top N baseline matches
            for (int i = 0; i < numberOfImages && i < distances.size(); ++i) {
                Mat image = imread(distances[i].second);
                imshow("Baseline Match " + to_string(i + 1), image); // Start from 1
            }
            cv::waitKey(0);
        }
        //Histogram Matching
        else if (choice == 2) {

            featureType = "HISTOGRAM MATCHING";

            std::cout << "Histogram Matching in Progres..." << std::endl;

            // Calculate distances for histogram features
            vector<pair<double, string>> distances;
            for (const auto& filename : filenames) {
                // Exclude the target image
                if (filename == TARGET_IMAGE) continue;

                // Read image
                Mat image = imread(filename);
                if (image.empty()) {
                    std::cout << "Error: Unable to read image " << filename << std::endl;
                    continue;
                }

                // Compute features
                Mat imageHistogramFeatures = colorHistogramMatching(image);

                // Compute distance
                double distance = norm(targetHistogramFeatures, imageHistogramFeatures, NORM_L2);
                distances.push_back(make_pair(distance, filename));
            }

            // Sort distances
            sort(distances.begin(), distances.end());

            // Display top N histogram matches
            std::cout << "Top " << numberOfImages << " histogram matches for the target image " << TARGET_IMAGE << ":" << std::endl;
            for (int i = 0; i < numberOfImages && i < distances.size(); ++i) {
                std::cout << distances[i].second << " (Distance: " << distances[i].first << ")" << std::endl;
            }

            // Display top N histogram matches
            for (int i = 0; i < numberOfImages && i < distances.size(); ++i) {
                Mat image = imread(distances[i].second);
                imshow("Histogram Match " + to_string(i + 1), image); // Start from 1
            }
            //cv::waitKey(0);
        }
        //Multi-histogram Matching
        else if (choice == 3) {

            featureType = "MULTI-HISTOGRAM MATCHING";

            std::cout << "Multi-Histogram Matching in Progres..." << std::endl;

            vector<pair<double, string>> distances;
            for (const auto& filename : filenames) {
                // Exclude the target image
                if (filename == TARGET_IMAGE) continue;

                // Read image
                Mat image = imread(filename);
                if (image.empty()) {
                    std::cout << "Error: Unable to read image " << filename << std::endl;
                    continue;
                }

                // Compute features
                Mat imageWholeImageHist = fullImageHistogram(image);
                Mat imageCenterRegionHist = centralHistogramFeatures(image);

                // Compute distance using custom distance metric
                double distance = computeCustomMetric(targetWholeImageHist, imageWholeImageHist) +
                    computeCustomMetric(targetCenterRegionHist, imageCenterRegionHist);
                distances.push_back(make_pair(distance, filename));
            }

            // Sort distances
            sort(distances.begin(), distances.end());

            // Display top N multi-histogram matches
            std::cout << "Top " << numberOfImages << " multi-histogram matches for the target image " << TARGET_IMAGE << ":" << std::endl;
            for (int i = 0; i < numberOfImages && i < distances.size(); ++i) {
                std::cout << distances[i].second << " (Distance: " << distances[i].first << ")" << std::endl;
            }

            // Display top N multi-histogram matches
            for (int i = 0; i < numberOfImages && i < distances.size(); ++i) {
                Mat image = imread(distances[i].second);
                imshow("Multi-Histogram Match " + to_string(i + 1), image); // Start from 1
            }
            //cv::waitKey(0);
        }
        //Texture and Color Matching
        else if (choice == 4) {

                featureType = "TEXTURE AND COLOR";
                std::cout << "Select Texture:" << std::endl;
                std::cout << "1. Sobel" << std::endl;
                std::cout << "2. Laws Filter" << std::endl;
                std::cout << "3. Gabor Filter" << std::endl;
                std::cout << "4. Fourier Transform" << std::endl;
                std::cout << "Enter choice (1-4): ";
                int option;
                std::cin >> option;

            if (option == 1) {
                //sobel
                std::cout << "Performing Sobel Filter" << std::endl;
                vector<pair<double, string>> distances;
                for (const auto& filename : filenames) {
                    // Exclude the target image
                    if (filename == TARGET_IMAGE) continue;

                    // Read image
                    Mat image = imread(filename);
                    if (image.empty()) {
                        std::cout << "Error: Unable to read image " << filename << std::endl;
                        continue;
                    }

                    // Compute features
                    Mat imageTextureFeatures = textureMatching(image);

                    // Compute distance
                    double distance = norm(targetTextureFeatures, imageTextureFeatures, NORM_L2);
                    distances.push_back(make_pair(distance, filename));
                }

                // Sort distances
                sort(distances.begin(), distances.end());

                // Display top N texture matches
                std::cout << "Top " << numberOfImages << " texture matches for the target image " << TARGET_IMAGE << ":" << std::endl;
                for (int i = 0; i < numberOfImages && i < distances.size(); ++i) {
                    std::cout << distances[i].second << " (Distance: " << distances[i].first << ")" << std::endl;
                }

                // Display top N texture matches
                for (int i = 0; i < numberOfImages && i < distances.size(); ++i) {
                    Mat image = imread(distances[i].second);
                    imshow("Texture Match " + to_string(i + 1), image); // Start from 1
                }
            }
            else if (option == 2) {
                //Laws Filter
                std::cout << "Performing Laws Filter" << std::endl;
                // Calculate distances for Laws' texture features
                vector<pair<double, string>> distances;
                for (const auto& filename : filenames) {
                    // Exclude the target image
                    if (filename == TARGET_IMAGE) continue;

                    // Read image
                    Mat image = imread(filename);
                    if (image.empty()) {
                        cout << "Error: Unable to read image " << filename << endl;
                        continue;
                    }

                    // Compute features
                    Mat imageLawsTextureFeatures = computeLawsTextureFeatures(image);

                    // Compute distance
                    double distance = norm(targetLawsTextureFeatures, imageLawsTextureFeatures, NORM_L2);
                    distances.push_back(make_pair(distance, filename));
                }

                // Sort distances
                sort(distances.begin(), distances.end());

                // Display top N Laws' texture matches
                cout << "Top " << numberOfImages << " Laws' texture matches for the target image " << TARGET_IMAGE << ":" << endl;
                for (int i = 0; i < numberOfImages && i < distances.size(); ++i) {
                    cout << distances[i].second << " (Distance: " << distances[i].first << ")" << endl;
                }

                // Display top N Laws' texture matches
                for (int i = 0; i < numberOfImages && i < distances.size(); ++i) {
                    Mat image = imread(distances[i].second);
                    imshow("Laws' Texture Match " + to_string(i + 1), image); // Start from 1
                }
                //cv::waitKey(0);
            }
            else if (option == 3) {
                //Gabor Filter
                std::cout << "Performing Gabor Filter" << std::endl;
                // Calculate distances for Gabor histogram features
                vector<pair<double, string>> distances;
                for (const auto& filename : filenames) {
                    // Exclude the target image
                    if (filename == TARGET_IMAGE) continue;

                    // Read image
                    Mat image = imread(filename);
                    if (image.empty()) {
                        cout << "Error: Unable to read image " << filename << endl;
                        continue;
                    }

                    // Compute features
                    Mat imageGaborHistogramFeatures = computeGaborHistogramFeatures(image);

                    // Compute distance
                    double distance = norm(targetGaborHistogramFeatures, imageGaborHistogramFeatures, NORM_L2);
                    distances.push_back(make_pair(distance, filename));
                }

                // Sort distances
                sort(distances.begin(), distances.end());

                // Display top N Gabor histogram matches
                cout << "Top " << numberOfImages << " Gabor histogram matches for the target image " << TARGET_IMAGE << ":" << endl;
                for (int i = 0; i < numberOfImages && i < distances.size(); ++i) {
                    cout << distances[i].second << " (Distance: " << distances[i].first << ")" << endl;
                }

                // Display top N Gabor histogram matches
                for (int i = 0; i < numberOfImages && i < distances.size(); ++i) {
                    Mat image = imread(distances[i].second);
                    imshow("Gabor Histogram Match " + to_string(i + 1), image); // Start from 1
                }
                //cv::waitKey(0);
            }
            else if (option == 4) {
                //Fourier Transforming
                std::cout << "Performing Fourier Transfomr" << std::endl;
                // Calculate distances for Fourier texture features
                vector<pair<double, string>> distances;
                for (const auto& filename : filenames) {
                    // Exclude the target image
                    if (filename == TARGET_IMAGE) continue;

                    // Read image
                    Mat image = imread(filename);
                    if (image.empty()) {
                        cout << "Error: Unable to read image " << filename << endl;
                        continue;
                    }

                    // Compute features
                    Mat imageFourierTextureFeatures = computeFourierTextureFeatures(image);

                    // Compute distance
                    double distance = norm(targetFourierTextureFeatures, imageFourierTextureFeatures, NORM_L2);
                    distances.push_back(make_pair(distance, filename));
                }

                // Sort distances
                sort(distances.begin(), distances.end());

                // Display top N Fourier texture matches

                cout << "Top " << numberOfImages << " Fourier texture matches for the target image " << TARGET_IMAGE << ":" << endl;
                for (int i = 0; i < numberOfImages && i < distances.size(); ++i) {
                    cout << distances[i].second << " (Distance: " << distances[i].first << ")" << endl;
                }

                // Display top N Fourier texture matches
                for (int i = 0; i < numberOfImages && i < distances.size(); ++i) {
                    Mat image = imread(distances[i].second);
                    imshow("Fourier Texture Match " + to_string(i + 1), image); // Start from 1
                }
               /* cv::waitKey(0);*/
            }

            
        }
        // DNN
        else if (choice == 5) {

            featureType = "DEEP NETWORK EMBEDDINGS";

            std::cout << "DNN Matching in Progres..." << std::endl;

            auto results = DNNBasedRetrieval(DIRECTORY, CSV_PATH, TARGET_IMAGE);

            // Display top 3 matches
            cout << "Top " + numberOfImages << " matches:" << endl;
            int count = 0;
            for (const auto& result : results) {
                if (count >= numberOfImages) break;
                cout << result.second << " (Distance: " << result.first << ")" << endl;
                Mat image = imread(DIRECTORY + "/" + result.second);
                imshow("DNN Match " + to_string(count + 1), image); // Start from 1
                count++;
            }
            
        }
        else if (choice == 6) {
            featureType = "CUSTOM DESIGN";
            std::cout << "Custom Design " << std::endl;
            auto results = dnn_image_retrieval(DIRECTORY, CSV_PATH, TARGET_IMAGE);

            // Display top 5 matches
            cout << "Top 5 matches:" << endl;
            int count = 0;
            for (const auto& result : results) {
                if (count >= 5) break;
                cout << result.second << " (Similarity: " << result.first << ")" << endl;
                Mat image = imread(DIRECTORY + "/" + result.second);
                imshow("DNN Match " + to_string(count + 1), image); // Start from 1
                count++;
            }

            cout << "Top 5 least matches:" << endl;
            count = 0;
            for (auto it = results.rbegin(); it != results.rend(); ++it) { // Reverse iterator
                if (count >= 5) break;
                cout << it->second << " (Similarity: " << it->first << ")" << endl;
                Mat image = imread(DIRECTORY + "/" + it->second);
                imshow("DNN Least Match " + to_string(count + 1), image); // Start from 1
                count++;
            }
            cv::waitKey();
        }
        // Face Detection
        else if (choice == 7) {
            
            auto results = dnn_image_retrieval(DIRECTORY, CSV_PATH, TARGET_IMAGE);

            // Sort results by similarity in descending order
            sort(results.begin(), results.end(), [](const pair<double, string>& a, const pair<double, string>& b) {
                return a.first > b.first; // Sort in descending order
                });

            cout << "\nAll most similar pictures:" << endl;

            // Use const reference in the loop to avoid copying
            for (const auto& result : results) {
                cout << result.second << " (Similarity: " << result.first << ")" << endl;

                // Use 'imread' to load image without copying the string
                Mat image = imread(DIRECTORY + "/" + result.second);

                if (!image.empty()) {
                    imshow("Most Similar", image);
                    waitKey(0); // Wait for a key press for each image
                }
                else {
                    cout << "Error: Image " << result.second << " could not be loaded." << endl;
                }
            }

        }
        waitKey(0);
        destroyAllWindows(); // Close all displayed windows before next iteration
    }

    std::cout << "Exiting program." << std::endl;
    return 0;
}