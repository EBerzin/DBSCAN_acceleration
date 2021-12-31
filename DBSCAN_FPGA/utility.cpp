// This file
#include "utility.h"
#include <math.h>
#include <iostream>
#include <stdio.h>

void print_platform_info(std::vector<cl::Platform>* PlatformList)
{
	uint num_platforms=PlatformList->size();
	std::cout << "Number of Platforms: "<<num_platforms<<"\n";
	//Grab Platform Info for each platform
	for (uint i=0; i<num_platforms; i++)
	{
		std::cout <<"Platform " << i <<": "<<PlatformList->at(i).getInfo<CL_PLATFORM_NAME>()<<"\n";
	}
	std::cout<<"\n";
}

uint get_platform_id_with_string(std::vector<cl::Platform>* PlatformList, const char * name)
{
	uint num_platforms=PlatformList->size();
	uint ret_value=-1;
	//Grab Platform Info for each platform
	for (uint i=0; i<num_platforms; i++)
	{
		std::basic_string<char> platform_name = PlatformList->at(i).getInfo<CL_PLATFORM_NAME>();
		if (platform_name.find(name)!=std::string::npos) {
				return i;
		}
	}
	return ret_value;
}

void print_device_info(std::vector<cl::Device>* DeviceList)
{
	uint num_devices=DeviceList->size();
	std::cout << "Number of Devices in Platform: "<<num_devices<<"\n";
	//Grab Device Info for each device
	for (uint i=0; i<num_devices; i++)
	{
		printf("Device Number: %d\n", i);
		std::cout << "Device Name: "<<DeviceList->at(i).getInfo<CL_DEVICE_NAME>()<<"\n";
		std::cout << "Is Device Available?: "<<DeviceList->at(i).getInfo<CL_DEVICE_AVAILABLE>()<<"\n";
		std::cout << "Device Max Compute Units: "<<DeviceList->at(i).getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>()<<"\n";
		std::cout << "Device Max Work Item Dimensions: "<<DeviceList->at(i).getInfo<CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS>()<<"\n";
		std::cout << "Device Max Work Group Size: "<<DeviceList->at(i).getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>()<<"\n";
		std::cout << "Device Max Frequency: "<<DeviceList->at(i).getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>()<<"\n";
		std::cout << "Device Max Mem Alloc Size: "<<DeviceList->at(i).getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>()<<"\n";
		std::cout << "Device Max Local Mem Size: "<<DeviceList->at(i).getInfo<CL_DEVICE_LOCAL_MEM_SIZE>()<<"\n";
		std::cout << "Device OpenCL Version: "<<DeviceList->at(i).getInfo<CL_DEVICE_OPENCL_C_VERSION>()<<"\n\n";
	}
}

void fill_generate(cl_float X[], cl_float Y[], cl_float Z[], cl_float LO, cl_float HI, size_t vectorSize)
{

	//Assigns randome number from LO to HI to all locatoin of X and Y
	for (uint i = 0; i < vectorSize; ++i) {
		X[i] =  LO + (cl_float)rand()/((cl_float)RAND_MAX/(HI-LO));
		Y[i] =  LO + (cl_float)rand()/((cl_float)RAND_MAX/(HI-LO));
	}
}

void checkErr(cl_int err, const char * name)
{
    if (err != CL_SUCCESS) {
        std::cerr << "ERROR: " << name
                 << " (" << err << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}

