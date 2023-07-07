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
typedef unsigned long long uint64_t;
typedef unsigned int uint32_t;
typedef unsigned short uint16_t;
typedef unsigned char uint8_t;
#else
#include <intrin.h>
#include <inttypes.h>
#include <math.h>
#ifndef DEVICE
#define DEVICE
#endif

struct alignas(4) uchar4
{
	unsigned char x;
	unsigned char y;
	unsigned char z;
	unsigned char w;
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
struct alignas(16) float4
{
	float x;
	float y;
	float z;
	float w;
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

#define PI 3.14159265358979323846264338327950288f
#define MAX_FLOAT 3.402823466e+38F


// -- Intrinsics --
#if defined( __CUDACC__ ) || defined( __HIPCC__ )
#define INTRIN_COS( x ) __cosf( ( x ) )
#define INTRIN_SIN( x ) __sinf( ( x ) )
#define INTRIN_SQRT( x ) __fsqrt_rn( ( x ) )
#define INTRIN_RSQRT( x ) __frsqrt_rn( ( x ) )
#define INTRIN_POW( x, y ) __expf( (y)*__logf( ( x ) ) )
#else
#define INTRIN_COS( x ) cosf( ( x ) )
#define INTRIN_SIN( x ) sinf( ( x ) )
#define INTRIN_SQRT( x ) sqrtf( ( x ) )
#define INTRIN_RSQRT( x ) ( 1.0f / sqrtf( ( x ) ) )
#define INTRIN_POW( x, y ) powf( ( x ), ( y ) )
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

template <class T>
DEVICE inline T ss_clamp( T x, T a, T b )
{
	return ss_min( ss_max( x, a ), b );
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
DEVICE inline float3 operator*( float3 a, float3 b )
{
	return { a.x * b.x, a.y * b.y, a.z * b.z };
}

DEVICE inline float3 operator/( float3 a, float b )
{
	return { a.x / b, a.y / b, a.z / b };
}
DEVICE inline float3 operator/( float3 a, float3 b )
{
	return { a.x / b.x, a.y / b.y, a.z / b.z };
}
#if defined( __CUDACC__ )
DEVICE inline float3& operator+=( float3& a, float3 b )
{
	a = a + b;
	return a;
}
DEVICE inline float3& operator-=( float3& a, float3 b )
{
	a = a - b;
	return a;
}
DEVICE inline float3& operator*=( float3& a, float3 b )
{
	a = a * b;
	return a;
}
#endif

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
DEVICE inline float3 fabs( float3 v )
{
	return { ss_abs( v.x ), ss_abs( v.y ), ss_abs( v.z ) };
}
DEVICE inline float maxElement( float a, float b, float c )
{
	return ss_max( ss_max( a, b ), c );
}
DEVICE inline float minElement( float a, float b, float c )
{
	return ss_min( ss_min( a, b ), c );
}
DEVICE inline float mix( float a, float b, float t )
{
	return a + ( b - a ) * t;
}
DEVICE inline float3 normalize( float3 v )
{
	return v * INTRIN_RSQRT( dot( v, v ) );
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

DEVICE inline uint32_t div_round_up( uint32_t val, uint32_t divisor )
{
	return ( val + divisor - 1 ) / divisor;
}
DEVICE inline uint32_t next_multiple( uint32_t val, uint32_t divisor )
{
	return div_round_up( val, divisor ) * divisor;
}