#include <stdlib.h>
#include <cstdlib>
#include <stdio.h>
#include <limits.h>
#include <sys/time.h>

#include "GPUDijkstraPlugin.h"
#include <fstream>
using namespace std;

const int NUMTHREADS = 128;
 

// Find the vertex with minimum distance value, from
// the set of vertices NOT yet included in updater[]
//the result would override updater[]  

int minDistance(int shortPath[], bool updater[], int V) {
   // Initialize min value
   int min = INT_MAX, min_index;

   for (int x = 0; x < V; x++)
     if (updater[x] == false && shortPath[x] <= min){
         min = shortPath[x], min_index = x;
}
	//printf("%d Min: ",min_index);
   return min_index;
}//end of method



void GPUDijkstraPlugin::input(std::string file) {
 inputfile = file;
 std::ifstream ifile(inputfile.c_str(), std::ios::in);
 while (!ifile.eof()) {
   std::string key, value;
   ifile >> key;
   ifile >> value;
   parameters[key] = value;
 }
 N = atoi(parameters["N"].c_str());
 int M = N * N;
     //allocate CPU variables in memory
     updater = (bool *)malloc(N*sizeof(bool));
     shortPath = (int *)malloc(N*sizeof(int));
     matrix = (long *)malloc(M*sizeof(long));
 std::ifstream myinput((std::string(PluginManager::prefix())+parameters["matrix"]).c_str(), std::ios::in);
 int i;
 for (i = 0; i < M; ++i) {
	int k;
	myinput >> k;
        matrix[i] = k;
 }
}

void GPUDijkstraPlugin::run() {
     // Initialize all distances as INFINITE(Maximum integer value) and stpSet[] as false
     for (int i = 0; i < N; i++){
        shortPath[i] = INT_MAX, updater[i] = false;
      }

     // Distance of source vertex from itself is always 0
     shortPath[0] = 0;

     //allocate GPU variables in memory
     cudaMalloc((void**) &d_updater,(N*sizeof(bool)));
     cudaMalloc((void**) &d_shortPath,(N*sizeof(int)));
     cudaMalloc((void**) &d_matrix,(N*N*sizeof(long)));

     //copying the matrix values to the GPU
     cudaMemcpy(d_matrix,matrix,N*N*sizeof(long),cudaMemcpyHostToDevice);


     // Find shortest path for all vertices
     for (int count = 0; count < N-1; count++) {
     // For every iteration, a minimum distance vertex is chosen from the set of vertices not yet processed
     //u is always equal to 0 in first iteration.
     int u = minDistance(shortPath, updater, N);	

	
     // Mark the picked vertex as processed
     updater[u] = true;

     //copy the updates to GPU so we can call updateDistance 	
     cudaMemcpy(d_updater,updater,N*sizeof(bool),cudaMemcpyHostToDevice);
     cudaMemcpy(d_shortPath,shortPath,N*sizeof(int),cudaMemcpyHostToDevice);

     //call updatedistance, 1st parameter=# of blocks we have,2nd= #of threads in that block
     updateDistance<<<N,NUMTHREADS>>>(d_updater,d_matrix,d_shortPath,u,N);
	
     //copy the updated values to the CPU
     cudaMemcpy(shortPath,d_shortPath,N*sizeof(int),cudaMemcpyDeviceToHost);
     cudaMemcpy(updater,d_updater,N*sizeof(bool),cudaMemcpyDeviceToHost);
      
     }//end of for

}

void GPUDijkstraPlugin::output(std::string file) {
     // print the constructed distance array
     std::ofstream outfile(file.c_str(), std::ios::out);
     outfile << "Vertex Distance from Source" << std::endl;
     for (int i = 0; i < N; i++)
	     outfile << i << " \t\t " << shortPath[i] << std::endl;

     //free memory space
     free(updater);
     free(shortPath);
     free(matrix);

     //free GPU variables
     cudaFree(d_shortPath);
     cudaFree(d_matrix);
     cudaFree(d_updater);    

}
PluginProxy<GPUDijkstraPlugin> GPUDijkstraPluginProxy = PluginProxy<GPUDijkstraPlugin>("GPUDijkstra", PluginManager::getInstance());

