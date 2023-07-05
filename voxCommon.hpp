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

	DEVICE uint64_t getMortonParent() const { return morton >> 3; }
};

DEVICE inline uint32_t hash( uint32_t x )
{
	x *= 0x9e3779b9u;
	x ^= x >> 16;
	return x;
}

struct OctreeNode
{
	uint8_t mask;
	uint32_t numberOfVoxels;
	uint32_t children[8];

	DEVICE uint32_t getHash() const
	{
		uint32_t h = hash( mask );
		for( int i = 0; i < 8; i++ )
		{
			h = h ^ hash( children[i] );
		}
		return h;
	}
	DEVICE bool operator==( const OctreeNode& rhs )
	{
		if( mask != rhs.mask )
		{
			return false;
		}
		for( int i = 0; i < 8; i++ )
		{
			if( children[i] != rhs.children[i] )
			{
				return false;
			}
		}
		return true;
	}

	DEVICE bool operator<( const OctreeNode& rhs ) const
	{
		// maybe fix this later
		if( mask != rhs.mask )
		{
			return mask < rhs.mask;
		}

		for( int i = 0; i < 8; i++ )
		{
			if( children[i] == rhs.children[i] )
			{
				continue;
			}
			return children[i] < rhs.children[i];
		}
		return false;
	}
};