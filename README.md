# DBSCAN_acceleration

This reposity contains all code written for my Fall 2021 JP: Accelerating DBSCAN for GNN-based Track Reconstruction
As denoted in the paper:

Implementation 2: DBSCAN_FPGA_CPU

Implementation 3: DBSCAN_FPGA

Implementation 4: GDBSCAN

The following are two example commands to run the code:

./DBSCAN --nsamples 20000 --nsectors 64 --min_samps 2 --eps 0.04 --precomputed --dist evt0_ptmin

./DBSCAN --nsamples 20000 --nsectors 64 --min_samps 2 --eps 0.04 --data evt0_ptmin



--nsamples: the maximum number of hits across all files to be run

--nsectors: the number of sectors/files in the event

--min_samps: an input parameter to DBSCAN, representing the minimum number of hits that constitutes a cluster

--eps: an input parameter to DBSCAN, representing the radius of a hit's neighborhood

--precomputed: if included, neighboring points are calculated according to a precomputed distance matrix, whose filename is specified following the "--dist" keyword.
	       if not included, neighboring points are calculated according to a set of 2D hit coordinates, whose filename is specified following the "--data" keyword.