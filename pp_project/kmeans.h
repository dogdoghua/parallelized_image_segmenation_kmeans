#ifndef KMEANS_H
#define KMEANS_H

#include <vector>
#include "define.h"

struct Center{
    double blueValue;
    double greenValue;
    double redValue;
};

using namespace cv;
using namespace std;

void createClustersInfo(uint8_t *, int clusters_number, vector<Scalar> & clustersCenters, vector<vector<Point>> & ptInClusters);

double computeColorDistance(Scalar pixel, Scalar clusterPixel);

void serial_findAssociatedCluster(uint8_t *, int clusters_number, vector<Scalar> & clustersCenters, vector<vector<Point>> & ptInClusters);
void omp_findAssociatedCluster(uint8_t *, int clusters_number, vector<Scalar> & clustersCenters, vector<vector<Point>> & ptInClusters);

double serial_adjustClusterCenters(uint8_t *, int clusters_number, vector<Scalar> & clustersCenters, 
                                vector<vector<Point>> & ptInClusters, double & oldCenter, vector<double> &serial_diffs);
double omp_adjustClusterCenters(uint8_t *, int clusters_number, vector<Scalar> & clustersCenters, 
                                vector<vector<Point>> & ptInClusters, double & oldCenter, vector<double> &omp_diffs);

void serial_applyFinalClusterToImage(uint8_t *, int clusters_number, vector<vector<Point>> ptInClusters, vector<Scalar> & clustersCenters);
void omp_applyFinalClusterToImage(uint8_t *, int clusters_number, vector<vector<Point>> ptInClusters, vector<Scalar> & clustersCenters);

void serial_kmeans(uint8_t *, int clusters_number, vector<vector<Point>> ptInClusters, vector<Scalar> & clustersCenters, vector<double> &serial_diffs);
void omp_kmeans(uint8_t *, int clusters_number, vector<vector<Point>> ptInClusters, vector<Scalar> & clustersCenters, vector<double> &omp_diffs, double &ompStart);

#endif