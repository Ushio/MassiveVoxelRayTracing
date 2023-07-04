#pragma once

#if defined( __CUDACC__ ) || defined( __HIPCC__ )
typedef unsigned long long uint64_t;
typedef unsigned int uint32_t;
typedef unsigned short uint16_t;
typedef unsigned char uint8_t;

#ifndef DEVICE
#define DEVICE __device__
#endif
#else
#include <inttypes.h>
#ifndef DEVICE
#define DEVICE
#endif
#endif

struct OctreeTask
{
	uint64_t morton;
	uint32_t child;
	uint32_t numberOfVoxels;
};