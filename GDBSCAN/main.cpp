#include <math.h>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <string>
#include <vector>
#include <stdlib.h>
#include <iostream>
#include <chrono>

#include "utility.h"
#include "CL/cl.hpp"

using namespace std;

int main(int argc, char *argv[]) {


  //Processing inputs
  cl_uint max_samples = 0;
  cl_uint nsectors = 0;
  cl_uint ndims = 2;
  cl_float eps = 0;
  cl_uint min_samps = 0;
  bool precomputed = false;
  string dist_matrix_fname = "";
  string data_fname = "";

  for(int i_arg = 1; i_arg < argc; i_arg++){
    string arg = argv[i_arg];
    if(arg == "--nsamples" && i_arg+1 < argc)        max_samples = atoi(argv[i_arg+1]);
    if(arg == "--nsectors" && i_arg+1 < argc)        nsectors = atoi(argv[i_arg+1]);
    if(arg == "--ndims" && i_arg+1 < argc)           ndims    = atoi(argv[i_arg+1]);
    if(arg == "--eps" && i_arg+1 < argc)             eps    = atof(argv[i_arg+1]);
    if(arg == "--min_samps" && i_arg+1 < argc)       min_samps   = atoi(argv[i_arg+1]);
    if(arg == "--precomputed")                       precomputed  = true;
    if(arg == "--dist" && i_arg+1 < argc)            dist_matrix_fname    = argv[i_arg+1];
    if(arg == "--data" && i_arg+1 < argc)            data_fname    = argv[i_arg+1];
  }

  // Validating inputs
  if(max_samples == 0) {
    cout << "ERROR: Enter the number of samples to process" << endl;
    return 1;
  }
  if(nsectors == 0) {
    cout << "ERROR: Enter the number of sectors to process" << endl;
    return 1;
  }
  if(eps == 0) {
    cout << "ERROR: Enter a value of epsilon" << endl;
    return 1;
  }
  if(min_samps == 0) {
    cout << "ERROR: Enter a value of min_samps" << endl;
    return 1;
  }
  if(precomputed && dist_matrix_fname == "") {
    cout << "ERROR: Enter a distance matrix filename to run precomputed metric" << endl;
    return 1;
  }
  if(!precomputed && data_fname == "") {
    cout << "ERROR: Enter a data filename" << endl;
    return 1;
  }
  
  cl_int err;

  // Setting up Platform Information
  std::vector<cl::Platform> PlatformList;
  err = cl::Platform::get(&PlatformList);
  checkErr(err, "Get Platform List");
  checkErr(PlatformList.size()>=1 ? CL_SUCCESS : -1, "cl::Platform::get");
  print_platform_info(&PlatformList);

  //Setup Device
  std::vector<cl::Device> DeviceList;
  err = PlatformList[0].getDevices(CL_DEVICE_TYPE_ALL, &DeviceList);
  checkErr(err, "Get Devices");
  print_device_info(&DeviceList);

  //Setting up Context
  cl::Context myContext(DeviceList, NULL, NULL, NULL, &err);
  checkErr(err, "Context Constructor");

  //Create Command Queue
  cl::CommandQueue myqueue(myContext, DeviceList[0], CL_QUEUE_PROFILING_ENABLE, &err);
  checkErr(err, "Queue Constructor");

  // Create buffers for input
  cl::Buffer bufX(myContext, CL_MEM_READ_ONLY, max_samples * sizeof(cl_float));
  cl::Buffer bufY(myContext, CL_MEM_READ_ONLY, max_samples * sizeof(cl_float));
  cl::Buffer bufDists(myContext, CL_MEM_READ_ONLY, max_samples * max_samples * sizeof(cl_float));


  //Create buffers for intermediate values
  cl::Buffer bufVa1(myContext, CL_MEM_READ_WRITE, max_samples * sizeof(cl_uint));
  cl::Buffer bufNeighbIdx(myContext, CL_MEM_READ_WRITE, max_samples * max_samples * sizeof(cl_uint));  
  cl::Buffer bufIsCore(myContext, CL_MEM_READ_WRITE, max_samples * sizeof(cl_uint));
  cl::Buffer bufVisited(myContext, CL_MEM_READ_WRITE, max_samples * sizeof(cl_uint));
  cl::Buffer bufLabels(myContext, CL_MEM_READ_WRITE, max_samples * sizeof(cl_int));

  cl_float X[max_samples]  __attribute__ ((aligned (64)));
  cl_float Y[max_samples]  __attribute__ ((aligned (64)));
  cl_float dist_matrix[max_samples * max_samples] __attribute__ ((aligned (64)));

  cl_uint va1[max_samples]  __attribute__ ((aligned (64)));
  cl_uint is_core[max_samples]  __attribute__ ((aligned (64)));
  cl_uint neighbIdx[max_samples * max_samples]  __attribute__ ((aligned (64)));
  cl_uint visited[max_samples]  __attribute__ ((aligned (64)));
  cl_int labels[max_samples]  __attribute__ ((aligned (64)));


  //Read input data
  vector<float> total_times;
  string ptmin[] = {"0p6", "0p7", "0p8", "0p9", "1", "1p5", "2"};

  for(int s = 0; s < nsectors; s++) {

    cl_uint vectorSize = 0;
    int i = 0;
    string inFileName;
    if(!precomputed) { 
      inFileName =  data_fname; //+ "_s" + to_string(s) + "_ptmin1GeV.txt";
    }
    else {
      inFileName = "/scratch/gpfs/eberzin/tracker_hits_dists/" + dist_matrix_fname + ptmin[s] + "GeV";
    }

    ifstream inFile;
    inFile.open(inFileName.c_str());
    if (inFile.is_open()) {
      int total_count = 0;
      while(!inFile.eof()) {
	i = vectorSize;
	labels[i] = -1;
	va1[i] = 0;
	is_core[i] = 0;
	visited[i] = 0;
	if(!precomputed) {
	  inFile >> X[i];
	  inFile >> Y[i];
	}
	else {
	  
	  string line;
          getline(inFile, line);
          istringstream line_data(line);

          string hit_dist;
          while(line_data >> hit_dist) {
            dist_matrix[total_count] = stof(hit_dist);
            total_count++;
          }
	}
	vectorSize++;
      }
      inFile.close(); // Close input file
    }
    else { //Error message
      cerr << "Can't find input file " << inFileName << endl;
    }
    vectorSize--;

    cout << "Read Data" << endl;
    

    //Write to buffers
    
    if(!precomputed) {
      err = myqueue.enqueueWriteBuffer(bufX, CL_TRUE, 0, vectorSize * sizeof(cl_float),X); checkErr(err, "WriteBuffer 1");
      err = myqueue.enqueueWriteBuffer(bufY, CL_TRUE, 0, vectorSize * sizeof(cl_float),Y); checkErr(err, "WriteBuffer 2");
    }
    else {
      err = myqueue.enqueueWriteBuffer(bufDists, CL_TRUE, 0, vectorSize * vectorSize * sizeof(cl_float),dist_matrix); checkErr(err, "WriteBuffer 1");
    }

    err = myqueue.enqueueWriteBuffer(bufVa1, CL_TRUE, 0, vectorSize * sizeof(cl_uint),va1); checkErr(err, "WriteBuffer 3");
    err = myqueue.enqueueWriteBuffer(bufIsCore, CL_TRUE, 0, vectorSize * sizeof(cl_uint),is_core); checkErr(err, "WriteBuffer 4");
    err = myqueue.enqueueWriteBuffer(bufNeighbIdx, CL_TRUE, 0, vectorSize * vectorSize * sizeof(cl_uint),neighbIdx); checkErr(err, "WriteBuffer 5");

    cout << "Wrote to buffers" << endl;
    
    //Creating kernels
    const char *kernel_name_make_graph_1;
    if (!precomputed) kernel_name_make_graph_1 = "radius_neighbors_step1";
    else kernel_name_make_graph_1 = "radius_neighbors_dists";
    const char *kernel_name_bfs = "breadth_first_search";

    // Creating Binaries
    std::ifstream aocx_stream("g_dbscan.aocx", std::ios::in|std::ios::binary);
    checkErr(aocx_stream.is_open() ? CL_SUCCESS:-1, "g_dbscan.aocx");
    std::string prog(std::istreambuf_iterator<char>(aocx_stream), (std::istreambuf_iterator<char>()));
    cl::Program::Binaries mybinaries(DeviceList.size(), std::make_pair(prog.c_str(), prog.length()));
    cout << "Created Binaries" << endl;
    

    // Create the Program from the AOCX file.
    cl::Program program(myContext, DeviceList, mybinaries, NULL, &err);
    checkErr(err, "Program Constructor");
    cout << "Created Program" << endl;

    // Build the program
    err= program.build(DeviceList);
    checkErr(err, "Build Program");


    // create the kernel
    cl::Kernel kernelGraph1(program, kernel_name_make_graph_1, &err);
    checkErr(err, "Kernel Creation 1 ");

    cl::Kernel kernelBFS(program, kernel_name_bfs, &err);
    checkErr(err, "Kernel Creation 2 ");

    chrono::time_point<std::chrono::high_resolution_clock> start = chrono::high_resolution_clock::now();


    //Setting Kernel Arguments for neighbor search
    if(!precomputed) {
      err = kernelGraph1.setArg(0, bufX); checkErr(err, "Arg 0");
      err = kernelGraph1.setArg(1, bufY); checkErr(err, "Arg 1");
    }
    else {
      err = kernelGraph1.setArg(0, bufDists); checkErr(err, "Arg 0");
    }
    err = kernelGraph1.setArg(2-precomputed, bufVa1); checkErr(err, "Arg 2");
    err = kernelGraph1.setArg(3-precomputed, bufNeighbIdx); checkErr(err, "Arg 4");
    err = kernelGraph1.setArg(4-precomputed, bufIsCore); checkErr(err, "Arg 5");
    err = kernelGraph1.setArg(5-precomputed, vectorSize); checkErr(err, "Arg 6");
    err = kernelGraph1.setArg(6-precomputed, eps); checkErr(err, "Arg 7");
    err = kernelGraph1.setArg(7-precomputed, min_samps); checkErr(err, "Arg 8");

    err = myqueue.enqueueNDRangeKernel(kernelGraph1, cl::NullRange, cl::NDRange(vectorSize), cl::NullRange, NULL);
    checkErr(err, "Kernel Execution");
    myqueue.finish();

    err= myqueue.enqueueReadBuffer(bufIsCore, CL_TRUE, 0, vectorSize * sizeof(cl_uint),is_core);

    cl::Buffer bufXa(myContext, CL_MEM_READ_WRITE, vectorSize * sizeof(cl_uint));
    cl::Buffer bufFa(myContext, CL_MEM_READ_WRITE, vectorSize * sizeof(cl_uint));

    //Setting the BFS kernel arguments that remain unchanged
    err = kernelBFS.setArg(0, bufVa1); checkErr(err, "Arg 0");
    err = kernelBFS.setArg(1, bufNeighbIdx); checkErr(err, "Arg 1");
    err = kernelBFS.setArg(4, bufIsCore); checkErr(err, "Arg 4");
    err = kernelBFS.setArg(5, vectorSize); checkErr(err, "Arg 5");

    int clusterID = 0;
    for(int i = 0; i < vectorSize; i++) {
      if (!visited[i] && is_core[i]) {
	visited[i] = 1;
	labels[i] = clusterID;
	
	//BFS
	cl_uint* Xa = new cl_uint[vectorSize];
	cl_uint* Fa = new cl_uint[vectorSize];
	memset(Xa, 0, sizeof(cl_uint)* vectorSize);
	memset(Fa, 0, sizeof(cl_uint)* vectorSize);
	Fa[i] = 1;
	err = myqueue.enqueueWriteBuffer(bufXa, CL_TRUE, 0, vectorSize * sizeof(cl_uint),Xa); checkErr(err, "WriteBuffer 1");
	err = myqueue.enqueueWriteBuffer(bufFa, CL_TRUE, 0, vectorSize * sizeof(cl_uint),Fa); checkErr(err, "WriteBuffer 2");
	int countFa = 1;

	err = kernelBFS.setArg(2, bufFa); checkErr(err, "Arg 2");
	err = kernelBFS.setArg(3, bufXa); checkErr(err, "Arg 3");

	while(countFa > 0) {
	  //Lauching parallel bfs
	  err = myqueue.enqueueNDRangeKernel(kernelBFS, cl::NullRange, cl::NDRange(vectorSize), cl::NullRange, NULL);
	  checkErr(err, "Kernel Execution");
	  myqueue.finish();

	  err= myqueue.enqueueReadBuffer(bufFa, CL_TRUE, 0, vectorSize * sizeof(cl_uint),Fa);
	  int empty = 1;
	  for(int k = 0; k < vectorSize; k++) {
	    if(Fa[k] == 1) {
	      empty = 0;
	      break;
	    }
	  }
	  if(empty) {countFa = 0;}
	}

	err= myqueue.enqueueReadBuffer(bufXa, CL_TRUE, 0, vectorSize * sizeof(cl_uint),Xa);
	for(int v = 0; v < vectorSize; v++) {
	  if(Xa[v] && !visited[v]) {
	    labels[v] = clusterID;
	    visited[v] = 1;
	  }
	}
	clusterID++;
	delete[] Xa;
	delete[] Fa;
      }

    }

    chrono::time_point<std::chrono::high_resolution_clock> stop = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::nanoseconds>(stop - start);

    cout << "TIME: " << duration.count() << endl;
    total_times.push_back(duration.count());

    err=myqueue.finish();
    checkErr(err, "Finish Queue");

    //Printing output labels to a file
    ofstream myfile("output_labels/" + dist_matrix_fname + ptmin[s] + "GeV_labels.txt");
    if (myfile.is_open())
      {
        for(int count = 0; count < vectorSize; count ++){
          myfile << labels[count] << endl;
        }
        myfile.close();
      }
    else cout << "Unable to open file";
  }

  //Printing output times to a file 
  ofstream output_file2("output_times/" + dist_matrix_fname + "_times_parallel.txt");
  ostream_iterator<float> output_iterator2(output_file2, "\n");
  std::copy(total_times.begin(), total_times.end(), output_iterator2);



  return 1;

}
