__kernel void radius_neighbors(__global float* restrict X,
                                __global float* restrict Y,
                                __global uint* restrict va1,
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

        va1[i] = nNeighb;
	if(nNeighb >= min_samps) {
		   is_core[i] = 1;
	}

}

__kernel void radius_neighbors_dists(__global float* restrict dist_matrix,
                                __global uint* restrict va1,
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

        va1[i] = nNeighb;
        if(nNeighb >= min_samps) {
                   is_core[i] = 1;
        }

}




__kernel void breadth_first_search(__global uint* restrict va1,
	      		            __global uint* restrict neighbIdx,
				    __global uint* restrict Fa,
				    __global uint* restrict Xa,
				     __global uint* restrict is_core,
				     uint N) {

	uint i = get_global_id(0);

	uint local_Fa = Fa[i];
	if (local_Fa) {
	   Fa[i] = 0;
	   Xa[i] = 1;
	
	   uint nNeighb = 0;
	   if (is_core[i]) {nNeighb = va1[i];}

	   for(int j = 0; j < nNeighb; j++) {
	   	   int nid = neighbIdx[N*i + j];
		   Fa[nid] = 1 - Xa[nid];
	   }

	}


}
