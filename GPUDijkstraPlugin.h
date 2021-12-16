#include "Plugin.h"
#include "PluginProxy.h"
#include <string>
#include <map>

class GPUDijkstraPlugin : public Plugin {

	public:
		void input(std::string file);
		void run();
		void output(std::string file);
	private:
                std::string inputfile;
		std::string outputfile;
		long* matrix;
		int N;
                std::map<std::string, std::string> parameters;
     //GPU
     int *d_shortPath;
     bool *d_updater;
     long* d_matrix;
     bool* updater;
     int* shortPath;

};

__global__ void updateDistance(bool* updater, long* matrix, int* shortPath, int u, int V){

int i = blockIdx.x * blockDim.x + threadIdx.x;

         // checking for the shortest distance and updating shortPath
         // Update shortPath[i] only when updater[i]!=true, there is an edge from  
         // the program will enter the if statement  as long as there are smaller nodes than the current one

         if(i<V){

    if ((!updater[i] && matrix[u*V+i] && shortPath[u] != INT_MAX && shortPath[u]+matrix[u*V+i] < shortPath[i]))
            shortPath[i] = shortPath[u] + matrix[u*V+i];

      }//end of if(big)
 
}//end of method
