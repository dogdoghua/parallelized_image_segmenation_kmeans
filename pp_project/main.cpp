#include <iostream>
#include <ctime>
#include <cmath>
#include <time.h>
#include <omp.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "kmeans.h"
#include "define.h"

extern FILE *f;
extern uint8_t *image, *origin_image, *knn_image;
extern int img_width, img_height, img_channels, img_size;

Mat mat_image, mat_origin_image;

using namespace cv;
using namespace std;

bool verfiy_result(vector<double> target, vector<double> golden){
    if(target.size() != golden.size()){
        printf("ERROR: target size differs from golden size\n");
        return false;
    }

    double epsilon = 0.000001;

    for(int i=1; i<target.size(); i++){
        if(abs(target[i] - golden[i]) > epsilon){
            printf("ERROR: target[%d] is not equal with golden, target: %.5f, golden: %.5f\n", i, target[i], golden[i]);
            return false;
        }
    }

    return true;
}

int main(int argc, char * argv[]) {

    string image_path = argv[1];
    //The number of cluster is the only parameter to choose
    int k = atoi(argv[2]);
    unsigned int MAX_THREAD = omp_get_max_threads();

    mat_origin_image = imread(image_path, IMREAD_COLOR);

    if(mat_origin_image.empty()){
        printf("Error opening image.\n");
        return -1;
    }

    mat_image = mat_origin_image;
    
    // uint8_t *origin_iamge
    origin_image = mat_origin_image.data;
    img_height = mat_origin_image.rows, img_width = mat_origin_image.cols, img_channels = mat_origin_image.channels();
    
    img_size = img_height * img_width * img_channels;

    // get memory
    image = (uint8_t *)calloc(img_size, sizeof(uint8_t));
    knn_image = (uint8_t *)calloc(img_size, sizeof(uint8_t));

    // copy from mat to dynic array;
    copy(origin_image, origin_image + img_size, image);

    printf("Image height: %d, Image width: %d\n", img_height, img_width);
 
    /*initial parameter*/
    vector<Scalar> clustersCenters, omp_clustersCenters;
    vector<vector<Point>> ptInClusters, omp_ptInClusters;

    /* create ramdom clusters centers and clusters vectors (serial & omp should be same)*/
    createClustersInfo(image, k, clustersCenters, ptInClusters);
    omp_clustersCenters = clustersCenters;
    omp_ptInClusters = ptInClusters;
    
    printf("---------serial start---------\n");
    double serialStart = omp_get_wtime(); 
    vector<double> serial_diffs;
    serial_kmeans(image, k, ptInClusters, clustersCenters, serial_diffs);
    
    double serialTime = omp_get_wtime() - serialStart;  
    printf("Serial time taken: %.2fs\n\n", serialTime);

    printf("Max system threads = %d\n", MAX_THREAD);

    // take serial result as golden result in a memory

    for (int i = 0 ; (1 << i) <= MAX_THREAD ; ++i ) {
        string f_name = "thread_" + to_string((1 << i))+".txt";
        f = fopen(f_name.c_str(), "w+");

        clustersCenters = omp_clustersCenters;
        ptInClusters = omp_ptInClusters;
        copy(origin_image, origin_image + img_size, image);

        fprintf(f, "---------openMP start---------\n");
        fprintf(f, "set system threads = %d\n", (1 << i));

        omp_set_num_threads((1 << i));

        double ompStart = omp_get_wtime();

        vector<double> omp_diffs;
        omp_kmeans(image, k, ptInClusters, clustersCenters, omp_diffs, ompStart);

        double ompTime = omp_get_wtime() - ompStart;

        if(!verfiy_result(omp_diffs, serial_diffs)){
            printf("OpenMP thread[%d] FAILED\n", 1<<i);
            break;
        }

        printf("OpenMP thread[%d] PASSED. Time taken: [%.2f]s, Speedup: [%.2f]\n", 1<<i, ompTime, serialTime / ompTime);

        fprintf(f, "OpenMP time taken: %.2fs\n", (omp_get_wtime() - ompStart));
        fclose(f);
    }

    free(image);
    free(knn_image);

    return 0;
}