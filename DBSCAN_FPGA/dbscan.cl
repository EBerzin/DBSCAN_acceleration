__kernel void radius_neighbors(__global float* restrict X,
                                __global float* restrict Y,
                                __global uint* restrict neighbCount,
                                __global uint* restrict neighbIdx,
                                __global uint* restrict is_core,
                                uint N, float eps, uint min_samps) {

        uint i = get_global_id(0);

        uint nNeighb = 0;
        float dist = 0;

        float local_X = X[i];
        float local_Y = Y[i];

        for(uint j = 0; j < N; j++) {
                 dist = (local_X - X[j]) * (local_X - X[j]) + (local_Y - Y[j]) * (local_Y - Y[j]);
                 if(dist <= eps * eps) {
                         neighbIdx[N*i + nNeighb] = j;
                         nNeighb++;
                 }
        }

        neighbCount[i] = nNeighb;
        if(nNeighb >= min_samps) {
                   is_core[i] = 1;
        }

}

__kernel void radius_neighbors_dists(__global float* restrict dist_matrix,
                                __global uint* restrict neighbCount,
                                __global uint* restrict neighbIdx,
                                __global uint* restrict is_core,
                                uint N, float eps, uint min_samps) {

        uint i = get_global_id(0);

        uint nNeighb = 0;
        float dist = 0;

        for(uint j = 0; j < N; j++) {
                 dist = dist_matrix[i*N + j];
                 if(dist <= eps) {
                         neighbIdx[N*i + nNeighb] = j;
                         nNeighb++;
                 }
        }

        neighbCount[i] = nNeighb;
        if(nNeighb >= min_samps) {
                   is_core[i] = 1;
        }

}


__kernel void label(__global uint* restrict neighbCount,
	      	__global uint* restrict neighbIdx,
		__global uint* restrict visited,
		__global int* restrict labels,
		__global uint* restrict is_core,
		uint N) {


	int clusterID = 0;
	int labelled[7000];
	int counter;
	for(int i = 0; i < N; i++) {
		if (!visited[i] && is_core[i]) {
		   visited[i] = 1;
		   labels[i] = clusterID;

		   counter = 0;
		   labelled[counter] = i;
		   counter++;

		   while(counter != 0) {
			int hit = labelled[counter-1];
			counter--;
			if(is_core[hit]) {
				int nNeighb = neighbCount[hit];
				for(int j = 0; j < nNeighb; j++) {
					int nidx = neighbIdx[hit*N + j];
					if (visited[nidx] == 0) {
					   visited[nidx] = 1;
					   labelled[counter] = nidx;
					   counter++;
					   labels[nidx] = clusterID;		   
					}
				}
			}
		   }
		   clusterID++;
		}
	}
}
