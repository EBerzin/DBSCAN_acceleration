#ifndef __UTILITY_H
#define __UTILITY_H

//#include <CL/cl2.hpp>
#include "CL/cl.hpp"
#include <vector>

void print_platform_info(std::vector<cl::Platform>* PlatformList);
uint get_platform_id_with_string(std::vector<cl::Platform>*, const char * name);
void print_device_info(std::vector<cl::Device>*);
void checkErr(cl_int err, const char * name);

#endif
