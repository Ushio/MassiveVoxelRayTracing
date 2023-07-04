#pragma once

#if defined( __CUDACC__ ) || defined( __HIPCC__ )
#define DEVICE __device__

DEVICE inline float ss_floor( float value )
{
	return floorf( value );
}
DEVICE inline float ss_ceil( float value )
{
	return ceilf( value );
}

#else
#include <intrin.h>
#define DEVICE

struct alignas( 8 ) int2
{
	int x;
	int y;
};

struct alignas( 16 ) int3
{
	int x;
	int y;
	int z;
};

struct alignas( 8 ) float2
{
	float x;
	float y;
};
struct alignas( 16 ) float3
{
	float x;
	float y;
	float z;
};
inline float ss_floor( float value )
{
	float d;
	_mm_store_ss( &d, _mm_floor_ss( _mm_setzero_ps(), _mm_set_ss( value ) ) );
	return d;
}
inline float ss_ceil( float value )
{
	float d;
	_mm_store_ss( &d, _mm_ceil_ss( _mm_setzero_ps(), _mm_set_ss( value ) ) );
	return d;
}

#endif

template <class T>
DEVICE inline T ss_max( T x, T y )
{
	return ( x < y ) ? y : x;
}
template <class T>
DEVICE inline T ss_min( T x, T y )
{
	return ( y < x ) ? y : x;
}
template <class T>
DEVICE inline T ss_abs( T x )
{
	return x >= T( 0 ) ? x : -x;
}

DEVICE inline float2 operator-( float2 a, float2 b )
{
	return { a.x - b.x, a.y - b.y };
}
DEVICE inline float2 operator+( float2 a, float2 b )
{
	return { a.x + b.x, a.y + b.y };
}
DEVICE inline float2 operator*( float2 a, float2 b )
{
	return { a.x * b.x, a.y * b.y };
}
DEVICE inline float2 operator*( float2 a, float b )
{
	return { a.x * b, a.y * b };
}

DEVICE inline float dot( float2 a, float2 b )
{
	return a.x * b.x + a.y * b.y;
}

DEVICE inline float3 operator-( float3 a, float3 b )
{
	return { a.x - b.x, a.y - b.y, a.z - b.z };
}
DEVICE inline float3 operator/( float3 a, float b )
{
	return { a.x / b, a.y / b, a.z / b };
}

DEVICE inline float3 fmaxf( float3 a, float3 b )
{
	return { ss_max( a.x, b.x ), ss_max( a.y, b.y ), ss_max( a.z, b.z ) };
}
DEVICE inline float3 fminf( float3 a, float3 b )
{
	return { ss_min( a.x, b.x ), ss_min( a.y, b.y ), ss_min( a.z, b.z ) };
}

DEVICE inline int3 maxi( int3 a, int3 b )
{
	return { ss_max( a.x, b.x ), ss_max( a.y, b.y ), ss_max( a.z, b.z ) };
}
DEVICE inline int3 mini( int3 a, int3 b )
{
	return { ss_min( a.x, b.x ), ss_min( a.y, b.y ), ss_min( a.z, b.z ) };
}

DEVICE inline float3 floorf( float3 v )
{
	return { ss_floor( v.x ), ss_floor( v.y ), ss_floor( v.z ) };
}

DEVICE inline float3 cross( float3 a, float3 b )
{
	return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x };
}


