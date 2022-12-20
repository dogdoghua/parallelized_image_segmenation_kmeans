
#include <iostream>
#include <ctime>
#include <cmath>
#include <time.h>
#include <omp.h>

#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "kmeans.h"
#include "define.h"

using namespace cv;
using namespace std;

extern FILE *f;
extern uint8_t *image, *origin_image, *knn_image;
extern int img_width, img_height, img_channels, img_size;

extern Mat mat_image, mat_origin_image;

// #pragma omp declare reduction (pushBack : vector<Point> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))

// randomly create clusters centers and clusters vectors
void createClustersInfo(uint8_t * img, int clusters_number, vector<Scalar> &clustersCenters, vector<vector<Point>> &ptInClusters){
    
    RNG random(cv::getTickCount());
    // parallel data race condition:(clustersCenters)
    
    // #pragma omp parallel for reduction(pushBack: clustersCenters) reduction(pushBack: ptInClusters)
    for(int k =0; k<clusters_number; k++){
            
        //get random pixel in image to initialize cluster center
        Point centerKPoint;
        centerKPoint.x = random.uniform(0, img_width);
        centerKPoint.y = random.uniform(0, img_height);

        int b_idx = get_offset(centerKPoint.y, centerKPoint.x, 0);
        int g_idx = get_offset(centerKPoint.y, centerKPoint.x, 1);
        int r_idx = get_offset(centerKPoint.y, centerKPoint.x, 2);

        //get color value of pixel and save it as a center
        Scalar centerK(*(img + b_idx), *(img + g_idx), *(img + r_idx));    
        
        clustersCenters.push_back(centerK);
            
        //create vector to store future associated pixel to each center
        vector<Point> ptInClusterK;
        ptInClusters.push_back(ptInClusterK);
    }
}

double computeColorDistance(Scalar pixel, Scalar clusterPixel){
    
    //use color difference to get distance to cluster
    double diffBlue = pixel.val[0] - clusterPixel[0];
    double diffGreen = pixel.val[1] - clusterPixel[1];
    double diffRed = pixel.val[2] - clusterPixel[2];
    //use euclidian distance to get distance
    double distance = sqrt(pow(diffBlue, 2) + pow(diffGreen,2) + pow(diffRed,2));
    
    return distance;
}

void serial_findAssociatedCluster(uint8_t * img, int clusters_number, vector<Scalar> & clustersCenters, vector<vector<Point>> & ptInClusters){
    
    // For each pixel, find closest cluster
    for (int r = 0 ; r < img_height ; ++r ){
        for (int c = 0; c < img_width ; ++c ){
            
            double minDistance = INFINITY;
            int closestClusterIndex = 0;

            int b_idx = get_offset(r, c, 0);
            int g_idx = get_offset(r, c, 1);
            int r_idx = get_offset(r, c, 2);

            Scalar pixel(*(img + b_idx), *(img + g_idx), *(img + r_idx));    
            
            for (int k = 0; k < clusters_number ; ++k ) {
                
                Scalar clusterPixel = clustersCenters[k];
                
                //use color difference to get distance to cluster
                double distance = computeColorDistance(pixel, clusterPixel);
               
                //update to closest cluster center
                if (distance < minDistance) {
                    minDistance = distance;
                    closestClusterIndex = k;
                }
            }
            //save pixel into associated cluster
            ptInClusters[closestClusterIndex].push_back(Point(c,r));
        }
    }
}

void omp_findAssociatedCluster(uint8_t * img, int clusters_number, Scalar* clustersCenters, vector<vector<Point>> & ptInClusters){
    // For each pixel, find closest cluster
    
    // dynamic allocate 2d array
    int **arr_ptInCluster = (int**)malloc(img_height * sizeof(int*));
    for(int i=0; i<img_height; ++i)
        arr_ptInCluster[i] = (int*)malloc(img_width * sizeof(int));

    #pragma omp parallel
    {
        uint8_t * local_img = img;
        #pragma omp for schedule(static)
        for (int r = 0 ; r < img_height ; ++r ) {
            for (int c = 0 ; c < img_width ; ++c ) {
                double minDistance = INFINITY;
                int closestClusterIndex = 0;

                int b_idx = get_offset(r, c, 0);
                int g_idx = get_offset(r, c, 1);
                int r_idx = get_offset(r, c, 2);

                Scalar pixel(*(local_img + b_idx), *(local_img + g_idx), *(local_img + r_idx)); 
                
                for(int k = 0; k<clusters_number; k++){
                        
                    Scalar clusterPixel = clustersCenters[k];
                        
                    //use color difference to get distance to cluster
                    double distance = computeColorDistance(pixel, clusterPixel);
                        
                    //update to closest cluster center
                    if(distance < minDistance){
                        minDistance = distance;
                        closestClusterIndex = k;
                    }
                }
                arr_ptInCluster[r][c] = closestClusterIndex;
            }
        }
    }

    for(int r = 0; r < img_height; ++r){
        for(int c = 0; c < img_width; ++c){
            int k=arr_ptInCluster[r][c];
            ptInClusters[k].push_back(Point(c, r));
        }
    }
    for(int i=0; i<img_height; ++i)
        free(arr_ptInCluster[i]);
    free(arr_ptInCluster);
}

double serial_adjustClusterCenters(uint8_t * img, int clusters_number, vector<Scalar> & clustersCenters, vector<vector<Point>> & ptInClusters, double & oldCenter, vector<double> &serial_diffs){
    
    double diffChange;
    double newCenter = 0;

    //adjust cluster center to mean of associated pixels
    for(int k =0; k<clusters_number; k++){
        
        vector<Point> ptInCluster = ptInClusters[k];
        double newBlue = 0, newGreen = 0, newRed = 0;
        
        //compute mean values for 3 channels
        for(int i=0; i<ptInCluster.size(); i++){

            int b_idx = get_offset(ptInCluster[i].y, ptInCluster[i].x, 0);
            int g_idx = get_offset(ptInCluster[i].y, ptInCluster[i].x, 1);
            int r_idx = get_offset(ptInCluster[i].y, ptInCluster[i].x, 2);         

            newBlue += *(img + b_idx);
            newGreen += *(img + g_idx);
            newRed += *(img + r_idx);
        }
        
        newBlue /= ptInCluster.size();
        newGreen /= ptInCluster.size();
        newRed /= ptInCluster.size();
        
        //assign new color value to cluster center
        Scalar newPixel(newBlue, newGreen, newRed);
        
        //compute distance between the old and new values
        
        newCenter += computeColorDistance(newPixel, clustersCenters[k]);

        clustersCenters[k] = newPixel;
    }
    
    newCenter /= clusters_number;
    
    //get difference between previous iteration change
    diffChange = abs(oldCenter - newCenter);
    serial_diffs.push_back(diffChange);
    // printf("center: %f \n", newCenter);
    cout << "diffChange is: " << diffChange << endl;
    oldCenter = newCenter;
    
    return diffChange;
}

double omp_adjustClusterCenters(uint8_t * img, int clusters_number, Scalar* clustersCenters, vector<vector<Point>> & ptInClusters, double & oldCenter, vector<double> &omp_diffs){

    double diffChange;
    double newCenter = 0;
    
    Scalar* clustersCenter_copy = clustersCenters;
    

    // vector<Scalar> clustersCenter_copy = clustersCenters;
    #pragma omp parallel //num_threads(omp_get_max_threads())
    {
        uint8_t * local_img = img;
        #pragma omp for schedule(static) 
        for (int k = 0; k < clusters_number; ++k ){
            // printf("thread[%d] take incharge of cluster[%d]\n", omp_get_thread_num(), k);
            vector<Point> ptInCluster = ptInClusters[k];   
            double threadBlue = 0, threadGreen = 0, threadRed = 0; 

            // # pragma omp parallel for reduction(+ : threadBlue, threadGreen, threadRed)
            int vecLen = ptInCluster.size();
            for(int i=0; i<vecLen; ++i){

                int b_idx = get_offset(ptInCluster[i].y, ptInCluster[i].x, 0);
                int g_idx = get_offset(ptInCluster[i].y, ptInCluster[i].x, 1);
                int r_idx = get_offset(ptInCluster[i].y, ptInCluster[i].x, 2);

                threadBlue += *(local_img + b_idx);
                threadGreen += *(local_img + g_idx);
                threadRed += *(local_img + r_idx);
            }
            
            threadBlue /= ptInCluster.size();
            threadGreen /= ptInCluster.size();
            threadRed /= ptInCluster.size();

            Scalar newPixel(threadBlue, threadGreen, threadRed);
           
            // #pragma omp critical
            // {
                newCenter += computeColorDistance(newPixel, clustersCenter_copy[k]);
            // }
            clustersCenters[k] = newPixel;
        }    
    }

    
    //adjust cluster center to mean of associated pixels
    newCenter /= clusters_number;

    //get difference between previous iteration change
    diffChange = abs(oldCenter - newCenter);
    omp_diffs.push_back(diffChange);
    fprintf(f, "diffChange is:%.5f\n", diffChange);
    oldCenter = newCenter;
    
    return diffChange;
}

void serial_applyFinalClusterToImage(uint8_t * img, int clusters_number, vector<vector<Point>> ptInClusters, vector<Scalar> & clustersCenters){
    
    srand(time(NULL));
    
    //assign random color to each cluster
    
    for(int k = 0 ; k < clusters_number ; ++k){
        vector<Point> ptInCluster = ptInClusters[k];
        Scalar randomColor = clustersCenters[k];
        // Scalar randomColor(rand() % 255,rand() % 255,rand() % 255);
        // for each pixel in cluster change color to fit cluster
        for(int i = 0; i < ptInCluster.size() ; ++i ){

            int b_idx = get_offset(ptInCluster[i].y, ptInCluster[i].x, 0);
            int g_idx = get_offset(ptInCluster[i].y, ptInCluster[i].x, 1);
            int r_idx = get_offset(ptInCluster[i].y, ptInCluster[i].x, 2);
            
            Scalar pixelColor = randomColor;

            *(img + b_idx) = pixelColor.val[0];
            *(img + g_idx) = pixelColor.val[1];
            *(img + r_idx) = pixelColor.val[2];
        }
    }
}

void omp_applyFinalClusterToImage(uint8_t * img, int clusters_number, vector<vector<Point>> ptInClusters, Scalar* clustersCenters){
    
    srand(time(NULL));

    int **arr_ptInCluster = (int**)malloc(img_height * sizeof(int*));
    for(int i=0; i<img_height; ++i)
        arr_ptInCluster[i] = (int*)malloc(img_width * sizeof(int));

    for(int k = 0; k < clusters_number; ++k){
        vector<Point> group = ptInClusters[k];
        for(int i=0; i<group.size(); ++i){
            arr_ptInCluster[group[i].y][group[i].x] = k;
        }
    }

    #pragma omp parallel
    {
        uint8_t * local_img = img;
        //assign random color to each cluster
        // #pragma omp for schedule(static)
        for(int k = 0; k < clusters_number; ++k){
            // vector<Point> ptInCluster = ptInClusters[k];
            Scalar randomColor = clustersCenters[k];
            
            #pragma omp for schedule(static) 
            for (int r = 0 ; r < img_height ; ++r ){
                for (int c = 0 ; c < img_width ; ++c ){
                    if(arr_ptInCluster[r][c]==k){
                        int b_idx = get_offset(r, c, 0);
                        int g_idx = get_offset(r, c, 1);
                        int r_idx = get_offset(r, c, 2);
                        
                        Scalar pixelColor = randomColor;

                        *(local_img + b_idx) = pixelColor.val[0];
                        *(local_img + g_idx) = pixelColor.val[1];
                        *(local_img + r_idx) = pixelColor.val[2];
                    }
                }
            }
        }
    }
    for(int i=0; i<img_height; ++i)
        free(arr_ptInCluster[i]);
    free(arr_ptInCluster);
    

}

void serial_kmeans(uint8_t * imgInput, int clusters_number, vector<vector<Point>> ptInClusters, vector<Scalar> & clustersCenters, vector<double> &serial_diffs) {
    //---------------------- K-MEANS -----------------------------
    //set up cluster center, cluster vector, and parameter to stop the iterations
    // vector<Scalar> clustersCenters;
    // vector< vector<Point> > ptInClusters;
    double threshold = 0.001;
    double oldCenter = INFINITY;
    double diffChange = oldCenter - 0;
    
    //iterate until cluster centers nearly stop moving (using threshold)
    int round = 0;
    while (diffChange > threshold) {
        
        //clear associated pixels for each cluster
        //can be parallel
        for (int k = 0 ; k < clusters_number ; ++k ) {
            ptInClusters[k].clear();
        }
        
        //find all closest pixel to cluster centers
        serial_findAssociatedCluster(imgInput, clusters_number, clustersCenters, ptInClusters);
        //recompute cluster centers values
        diffChange = serial_adjustClusterCenters(imgInput, clusters_number, clustersCenters, ptInClusters, oldCenter, serial_diffs);

        std::copy(imgInput, imgInput + img_size, knn_image);

        serial_applyFinalClusterToImage(knn_image, clusters_number, ptInClusters, clustersCenters);
    
        mat_image.data = knn_image;

        //imshow("Segmentation", mat_image);
        cv::imwrite("./segement/serial/KMeansSegmentation_Round_"+to_string(round)+".jpg", mat_image);
        ++round;
        // cv::waitKey(50);
    }
}

void omp_kmeans(uint8_t * imgInput, int clusters_number, vector<vector<Point>> ptInClusters, vector<Scalar> & clustersCenters, vector<double> &omp_diffs, double &ompStart) {
    //---------------------- K-MEANS -----------------------------
    double threshold = 0.001;
    double oldCenter = INFINITY;
    double diffChange = oldCenter - 0;
    
    Scalar *arr_clustersCenters = (Scalar*)malloc(clusters_number * sizeof(Scalar));
    for(int i = 0; i < clusters_number; ++i) { 
        arr_clustersCenters[i] = clustersCenters[i];
    }
    double findTime=0, adjustTime=0, applyTime=0, startTime=0, a=0, b=0, c=0;

    //iterate until cluster centers nearly stop moving (using threshold)
    int round = 0;
    while ( diffChange > threshold) {
        //clear associated pixels for each cluster
        //can be parallel
        
        #pragma omp parallel for schedule(static)
        for (int k = 0 ; k < clusters_number ; ++k ){
            ptInClusters[k].clear();
        }
        //find all closest pixel to cluster centers
        double findStart=omp_get_wtime();
        omp_findAssociatedCluster(imgInput, clusters_number, arr_clustersCenters, ptInClusters);
        a=omp_get_wtime();
        
        //recompute cluster centers values
        double adjustStart=omp_get_wtime();
        diffChange = omp_adjustClusterCenters(imgInput, clusters_number, arr_clustersCenters, ptInClusters, oldCenter, omp_diffs);
        b=omp_get_wtime();

        std::copy(imgInput, imgInput + img_size, knn_image);
        double applyStart=omp_get_wtime();
        omp_applyFinalClusterToImage(knn_image, clusters_number, ptInClusters, arr_clustersCenters);
        c=omp_get_wtime();

        mat_image.data = knn_image;

        //imshow("Segmentation", mat_image);
        cv::imwrite("./segement/omp/KMeansSegmentation_Round_"+to_string(round)+".jpg", mat_image);
        ++round;
        // cv::waitKey(50);
        findTime += a-findStart;
        adjustTime += b-adjustStart;
        applyTime += c-applyStart;
    }
    double totalTime=omp_get_wtime()-ompStart;
    printf("Proportion of omp_findAssociatedCluster is: %.3f%%\n", findTime/totalTime * 100.0);
    printf("Proportion of omp_adjustClusterCenters is: %.3f%%\n", adjustTime/totalTime * 100.0);
    printf("Proportion of omp_applyFinalClusterToImage is: %.3f%%\n", applyTime/totalTime * 100.0);
    printf("Proportion of total is: %.3fs\n", totalTime);
    // printf("parallel omp_findAssociatedCluster time taken [%.4f]s\n", findTime);


    free(arr_clustersCenters);
}