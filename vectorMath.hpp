#pragma once

#if defined( __CUDACC__ ) || defined( __HIPCC__ )
#ifndef DEVICE
#define DEVICE __device__
#endif
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

#ifndef DEVICE
#define DEVICE
#endif
struct alignas(4) uchar4
{
	uint8_t x;
	uint8_t y;
	uint8_t z;
	uint8_t w;
};
struct int2
{
	int x;
	int y;
};

struct int3
{
	int x;
	int y;
	int z;
};

struct float2
{
	float x;
	float y;
};
struct float3
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
DEVICE inline float dot( float3 a, float3 b )
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}
DEVICE inline float3 operator-( float3 a )
{
	return { -a.x, -a.y, -a.z };
}
DEVICE inline float3 operator+( float3 a, float3 b )
{
	return { a.x + b.x, a.y + b.y, a.z + b.z };
}
DEVICE inline float3 operator-( float3 a, float3 b )
{
	return { a.x - b.x, a.y - b.y, a.z - b.z };
}
DEVICE inline float3 operator*( float3 a, float b )
{
	return { a.x * b, a.y * b, a.z * b };
}
DEVICE inline float3 operator*( float a, float3 b )
{
	return { a * b.x, a * b.y, a * b.z };
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

DEVICE inline float3 closestBarycentricCoordinateOnTriangle( float3 v0, float3 v1, float3 v2, float3 P )
{
	float3 d0 = v0 - P;
	float3 d1 = v1 - P;
	float3 d2 = v2 - P;
	float3 e0 = v2 - v0;
	float3 e1 = v0 - v1;
	float3 e2 = v1 - v2;
	float3 Ng = cross( e2, e0 );

	// bc inside the triangle
	// barycentric coordinate from tetrahedron volumes
	float U = dot( cross( d2, d0 ), Ng );
	float V = dot( cross( d0, d1 ), Ng );
	float W = dot( cross( d1, d2 ), Ng );

	// bc outside the triangle
	if( U < 0.0f )
	{
		V = dot( -d0, e0 );
		W = dot( d2, e0 );
	}
	else if( V < 0.0f )
	{
		W = dot( -d1, e1 );
		U = dot( d0, e1 );
	}
	else if( W < 0.0f )
	{
		U = dot( -d2, e2 );
		V = dot( d1, e2 );
	}

	float3 bc = fmaxf( float3{ 0.0f, 0.0f, 0.0f }, { U, V, W } );
	return bc / ( bc.x + bc.y + bc.z );
}