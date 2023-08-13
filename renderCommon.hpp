#pragma once

#include "vectorMath.hpp"

#if defined( __CUDACC__ ) || defined( __HIPCC__ )

#else
#include <glm/glm.hpp>
#include "Orochi/Orochi.h"
#include "hipUtil.hpp"
#endif

#define RENDER_NUMBER_OF_THREAD 256
#define USE_PMJ 1


class CameraPinhole
{
public:
#if !defined( __CUDACC__ ) && !defined( __HIPCC__ )
	void initFromPerspective( glm::mat4 viewMatrix, glm::mat4 projMatrix, float focus, float lensR )
	{
		glm::mat3 vT = glm::transpose( glm::mat3( viewMatrix ) );
		m_front = { -vT[2].x, -vT[2].y, -vT[2].z };
		m_up = { vT[1].x, vT[1].y, vT[1].z };
		m_right = { vT[0].x, vT[0].y, vT[0].z };

		glm::vec3 m = vT * glm::vec3( viewMatrix[3] );
		m_o = { -m.x, -m.y, -m.z };

		m_tanHthetaY = 1.0f / projMatrix[1][1];

		m_lensR = lensR;
		m_focus = focus;
	}
#endif
	DEVICE void shoot( float3* ro, float3* rd, int x, int y, float xoffsetInPixel, float yoffsetInPixel, int imageWidth, int imageHeight ) const
	{
		float xf = ( x + xoffsetInPixel ) / imageWidth;
		float yf = ( y + yoffsetInPixel ) / imageHeight;

		float3 d =
			m_right * mix( -m_tanHthetaY, m_tanHthetaY, xf ) * imageWidth / imageHeight +
			m_up * mix( m_tanHthetaY, -m_tanHthetaY, yf ) +
			m_front;

		*ro = m_o;
		*rd = d;
	}
	DEVICE void shootThinLens( float3* ro, float3* rd, int x, int y, float xoffsetInPixel, float yoffsetInPixel, int imageWidth, int imageHeight, float u0, float u1 ) const
	{
		float xf = ( x + xoffsetInPixel ) / imageWidth;
		float yf = ( y + yoffsetInPixel ) / imageHeight;

		// local coords: 
		float3 focalP = {
			m_focus * mix( -m_tanHthetaY, m_tanHthetaY, xf ) * imageWidth / imageHeight,
			m_focus * mix( m_tanHthetaY, -m_tanHthetaY, yf ),
			m_focus
		};
		float3 lensP = {
			mix( -m_lensR, m_lensR, u0 ),
			mix( -m_lensR, m_lensR, u1 ),
			0.0f
		};
		float3 dir = focalP - lensP;

		// to world:
		float3 d =
			m_right * dir.x +
			m_up * dir.y +
			m_front * dir.z;
		*rd = d;
		*ro = m_o + m_right * lensP.x + m_up * lensP.y + m_front * lensP.z;
	}

	float3 m_o;
	float3 m_front;
	float3 m_up;
	float3 m_right;
	float m_tanHthetaY;
	float m_lensR;
	float m_focus;
};

struct PCG32
{
	uint64_t state;
	uint64_t inc;

	DEVICE void setup( uint64_t seed, uint64_t stream )
	{
		state = 0;
		inc = stream * 2 + 1;

		nextU32();
		state += seed;
		nextU32();
	}
	DEVICE uint32_t nextU32()
	{
		uint64_t oldstate = state;
		// Advance internal state
		state = oldstate * 6364136223846793005ULL + inc;
		// Calculate output function (XSH RR), uses old state for max ILP
		uint32_t xorshifted = ( ( oldstate >> 18u ) ^ oldstate ) >> 27u;
		uint32_t rot = oldstate >> 59u;
		return ( xorshifted >> rot ) | ( xorshifted << ( ( -rot ) & 31 ) );
	}
};

DEVICE inline float uniformf( uint32_t x )
{
	uint32_t bits = ( x >> 9 ) | 0x3f800000;
	float value = *reinterpret_cast<float*>( &bits ) - 1.0f;
	return value;
}

DEVICE inline void GetOrthonormalBasis( float3 zaxis, float3* xaxis, float3* yaxis )
{
	const float sign = copysignf( 1.0f, zaxis.z );
	const float a = -1.0f / ( sign + zaxis.z );
	const float b = zaxis.x * zaxis.y * a;
	*xaxis = float3{ 1.0f + sign * zaxis.x * zaxis.x * a, sign * b, -sign * zaxis.x };
	*yaxis = float3{ b, sign + zaxis.y * zaxis.y * a, -zaxis.y };
}

// PDF( Lambertian BRDF ) = cos( theta ) / PI
// Sampling:
//   Lambertian BRDF = R / PI
//   Lambertian BRDF * cos( theta ) / PDF( Lambertian BRDF )
//       = ( R / PI ) * cos( theta ) * ( PI / cos( theta ))
//       = R
DEVICE inline float3 sampleLambertian( float a, float b, const float3& Ng )
{
	float r = INTRIN_SQRT( a );
	float theta = b * PI * 2.0f;

	// uniform in xy circle, a = r * r
	float x = r * INTRIN_COS( theta );
	float y = r * INTRIN_SIN( theta );

	// unproject to hemisphere
	float z = INTRIN_SQRT( ss_max( 1.0f - a, 0.0f ) );

	// local to global
	float3 xaxis;
	float3 yaxis;
	GetOrthonormalBasis( Ng, &xaxis, &yaxis );
	return xaxis * x + yaxis * y + Ng * z;
}

DEVICE inline float3 linearReflectance( uchar4 color )
{
	return {
		INTRIN_POW( (float)color.x / 255.0f, 2.2f ),
		INTRIN_POW( (float)color.y / 255.0f, 2.2f ),
		INTRIN_POW( (float)color.z / 255.0f, 2.2f ) };
}
DEVICE inline float3 rawReflectance( uchar4 color )
{
	return {
		(float)color.x / 255.0f,
		(float)color.y / 255.0f,
		(float)color.z / 255.0f };
}

template <class T>
DEVICE inline float luminance( T color )
{
	return 0.2126f * color.x + 0.7152f * color.y + 0.0722f * color.z;
}

// forward: +x, up: +y
DEVICE inline float2 getSpherical( float3 n )
{
	float phi = atan2f( n.z, n.x ) + PI;
	float theta = atan2f( INTRIN_SQRT( n.x * n.x + n.z * n.z ), n.y );
	return { phi / ( PI * 2.0f ), theta / PI };
}

template <class F>
DEVICE inline int upper_bound_f( F f, int n, float b )
{
	int i = 0;
	int j = n;

	while( i < j )
	{
		const int m = ( i + j ) / 2;
		const float value = f( m );
		if( value <= b )
		{
			i = m + 1;
		}
		else
		{
			j = m;
		}
	}
	return i;
}

struct HDRI
{
#if !defined( __CUDACC__ ) && !defined( __HIPCC__ )
	HDRI()
	{
		for( int i = 0; i < 6; i++ )
		{
			m_sats[i] = 0;
		}
	}
	void load( float* ptr, int width, int height, Shader* voxKernel, oroStream stream )
	{
		m_width = width;
		m_height = height;
		if( m_pixels )
		{
			oroFree( (oroDeviceptr)m_pixels );
		}
		if( m_sat )
		{
			oroFree( (oroDeviceptr)m_sat );
		}
		for( int i = 0; i < 6; i++ )
		{
			if( m_sats[i] )
			{
				oroFree( (oroDeviceptr)m_sats[i] );
				m_sats[i] = 0;
			}
		}
#define SAT_BLOCK_SIZE 512

		uint64_t pixelBytes = sizeof( float4 ) * m_width * m_height;
		oroMalloc( (oroDeviceptr*)&m_pixels, pixelBytes );
		oroMemcpyHtoDAsync( (oroDeviceptr)m_pixels, ptr, pixelBytes, stream );

		uint64_t satBytes = sizeof( uint32_t ) * m_width * m_height;
		oroMalloc( (oroDeviceptr*)&m_sat, satBytes );

		Buffer satF64( sizeof( double ) * m_width * m_height );
		{
			ShaderArgument args;
			args.add( m_pixels );
			args.add<int2>( { m_width, m_height } );
			args.add( satF64.data() );
			args.add( 0 );
			args.add( float3{ 0.0f, 0.0f, 0.0f } );
			voxKernel->launch( "HDRIstoreImportance", args, 
				div_round_up64( m_width, 8 ), div_round_up64( m_height, 8 ), 1, 
				8, 8, 1, stream );
		}
		{
			ShaderArgument args;
			args.add<int2>( { m_width, m_height } );
			args.add( satF64.data() );
			voxKernel->launch( "buildSATh", args, m_height, 1, 1, SAT_BLOCK_SIZE, 1, 1, stream );
			voxKernel->launch( "buildSATv", args, m_width, 1, 1, SAT_BLOCK_SIZE, 1, 1, stream );
		}
		{
			ShaderArgument args;
			args.add( m_sat );
			args.add( satF64.data() );
			args.add( m_width * m_height );
			voxKernel->launch( "buildSAT2u32", args, div_round_up64( m_width * m_height, 64 ), 1, 1, 64, 1, 1, stream );
		}

		float3 axisList[6] = {
			{ +1.0f, 0.0f, 0.0f },
			{ -1.0f, 0.0f, 0.0f },
			{ 0.0f, +1.0f, 0.0f },
			{ 0.0f, -1.0f, 0.0f },
			{ 0.0f, 0.0f, +1.0f },
			{ 0.0f, 0.0f, -1.0f },
		};
		for( int i = 0; i < 6; i++ )
		{
			if( m_sats[i] )
			{
				oroFree( (oroDeviceptr)m_sats[i] );
			}
			oroMalloc( (oroDeviceptr*)&m_sats[i], satBytes );

			{
				ShaderArgument args;
				args.add( m_pixels );
				args.add<int2>( { m_width, m_height } );
				args.add( satF64.data() );
				args.add( 1 );
				args.add( axisList[i] );
				voxKernel->launch( "HDRIstoreImportance", args,
								   div_round_up64( m_width, 8 ), div_round_up64( m_height, 8 ), 1,
								   8, 8, 1, stream );
			}
			{
				ShaderArgument args;
				args.add<int2>( { m_width, m_height } );
				args.add( satF64.data() );
				voxKernel->launch( "buildSATh", args, m_height, 1, 1, SAT_BLOCK_SIZE, 1, 1, stream );
				voxKernel->launch( "buildSATv", args, m_width, 1, 1, SAT_BLOCK_SIZE, 1, 1, stream );
			}
			{
				ShaderArgument args;
				args.add( m_sats[i] );
				args.add( satF64.data() );
				args.add( m_width * m_height );
				voxKernel->launch( "buildSAT2u32", args, div_round_up64( m_width * m_height, 64 ), 1, 1, 64, 1, 1, stream );
			}
		}

		oroStreamSynchronize( stream );
	}
	void loadPrimary( float* ptr, oroStream stream )
	{
		if( m_pixelsPrimary )
		{
			oroFree( (oroDeviceptr)m_pixelsPrimary );
		}

		uint64_t pixelBytes = sizeof( float4 ) * m_width * m_height;
		oroMalloc( (oroDeviceptr*)&m_pixelsPrimary, pixelBytes );
		oroMemcpyHtoDAsync( (oroDeviceptr)m_pixelsPrimary, ptr, pixelBytes, stream );
	}
	void cleanUp()
	{
		if( m_pixels )
		{
			oroFree( (oroDeviceptr)m_pixels );
			m_pixels = 0;
		}
		if( m_sat )
		{
			oroFree( (oroDeviceptr)m_sat );
			m_sat = 0;
		}
		for( int i = 0; i < 6; i++ )
		{
			if( m_sats[i] )
			{
				oroFree( (oroDeviceptr)m_sats[i] );
				m_sats[i] = 0;
			}
		}
		if( m_pixelsPrimary )
		{
			oroFree( (oroDeviceptr)m_pixelsPrimary );
			m_pixelsPrimary = 0;
		}
	}
#else
	DEVICE float3 sampleNearest( float3 direction, bool isPrimary ) const
	{
		float2 uv = getSpherical( direction );
		int x = (int)ss_clamp( uv.x * m_width, 0.0f, (float)( m_width - 1.0f ) );
		int y = (int)ss_clamp( uv.y * m_height, 0.0f, (float)( m_height - 1.0f ) );
		uint64_t index = (uint64_t)y * m_width + x;
		float4 c = ( isPrimary && m_pixelsPrimary ) ? m_pixelsPrimary[index] : m_pixels[index];
		return float3 { c.x, c.y, c.z } * m_scale;
	}
#endif
	DEVICE void importanceSample( float3* direction, float3* L, float* srPDF, float3 N, bool axisAligned, float u0, float u1, float u2, float u3 ) const
	{
		uint32_t* sat = m_sat;

		if( axisAligned )
		{
			const float k = 0.8f;
			if( k < N.x )
			{
				sat = m_sats[0];
			}
			else if( N.x < -k )
			{
				sat = m_sats[1];
			}
			else if( k < N.y )
			{
				sat = m_sats[2];
			}
			else if( N.y < -k )
			{
				sat = m_sats[3];
			}
			else if( k < N.z )
			{
				sat = m_sats[4];
			}
			else if( N.z < -k )
			{
				sat = m_sats[5];
			}
		}
		
		uint32_t X = upper_bound_f( [this, sat]( int i ){ return (float)getPrefixSumExclusiveH( sat, i ) / (float)0xFFFFFFFFu; }, m_width, u0 ) - 1;

		// H prefix sum range is not 0 to 0xFFFFFFFF need to adjust.
		uint32_t vol = getPrefixSumExclusiveH( sat, X + 1 ) - getPrefixSumExclusiveH( sat, X );
		uint32_t Y = upper_bound_f( [this, sat, X, vol]( int i ){ return  (float)getPrefixSumExclusiveV( sat, X, i ) / (float)vol; }, m_height, u1 ) - 1;

		float pSelection = (float)getCount( sat, X, Y ) / (float)0xFFFFFFFF;

		float dTheta = PI / (float)m_height;
		float dPhi = 2.0f * PI / (float)m_width;

		float theta = Y * dTheta;

		// dH = cos( theta ) - cos( theta + dTheta )
		//    = 2 sin( dTheta / 2 ) sin( dTheta / 2 + theta )
		float dH = 2.0f * INTRIN_SIN( dTheta * 0.5f ) * INTRIN_SIN( dTheta * 0.5f + theta );
		float dW = dPhi;
		float sr = dH * dW;
		
		float sY = mix( INTRIN_COS( theta ), INTRIN_COS( theta + dTheta ), u2 );

		float phi = dPhi * ( (float)X + u3 ) + PI;
		float sX = INTRIN_COS( phi );
		float sZ = INTRIN_SIN( phi );

		float sinTheta = INTRIN_SQRT( ss_max( 1.0f - sY * sY, 0.0f ) );
		*direction = {
			sX * sinTheta,
			sY,
			sZ * sinTheta,
		};
		*srPDF = pSelection / sr;

		float4 color = m_pixels[Y * m_width + X];
		*L = float3{ color.x, color.y, color.z } * m_scale;
	}

	DEVICE uint32_t getPrefixSumExclusiveH( const uint32_t* sat, uint32_t x ) const
	{
		if( x <= 0 )
		{
			return 0;
		}
		return sat[m_width * ( m_height - 1 ) + x - 1];
	}
	DEVICE uint32_t getPrefixSumExclusiveV( const uint32_t* sat, uint32_t x, uint32_t y ) const
	{
		if( y <= 0 )
		{
			return 0;
		}

		uint32_t s0 = x <= 0 ? 0 : sat[m_width * ( y - 1 ) + ( x - 1 )];
		uint32_t s1 = sat[m_width * ( y - 1 ) + x];
		return s1 - s0;
	}
	DEVICE uint32_t getCount( const uint32_t* sat, uint32_t x, uint32_t y ) const
	{
		// AB
		// CD
		uint32_t a = ( x <= 0 || y <= 0 ) ? 0 : sat[m_width * ( y - 1 ) + ( x - 1 )];
		uint32_t b = ( y <= 0 ) ? 0 : sat[m_width * ( y - 1 ) + x];
		uint32_t c = ( x <= 0 ) ? 0 : sat[m_width * y + ( x - 1 )];
		uint32_t d = sat[m_width * y + x];
		return ( d - b ) + ( a - c );
	}

	DEVICE bool isEnabled() const
	{
		return 0.0f < m_scale;
	}

	float4* m_pixels = 0;
	float4* m_pixelsPrimary = 0;
	uint32_t* m_sat = 0;
	uint32_t* m_sats[6]; // +x, -x, +y, -y, +z, -z
	int m_width = 0;
	int m_height = 0;
	float m_scale = 1.75f;
};

#if !defined( __CUDACC__ ) && !defined( __HIPCC__ )
inline uint32_t gcd( uint32_t a, uint32_t b )
{
	if( b == 0 )
		return a;
	return gcd( b, a % b );
}
#endif

// from "Bandwidth-Optimal Random Shuffling for GPUs"
struct LCGShuffler
{
	uint32_t a = 1;
	uint32_t c = 0;
	uint32_t n = 0;

	// ( a * x + c ) mod n
	DEVICE uint32_t operator()( uint32_t i ) const
	{
		return ( static_cast<uint64_t>( i ) * a + c ) % n;
	}
#if !defined( __CUDACC__ ) && !defined( __HIPCC__ )
	// return true if succeeded
	bool tryInit( uint32_t r0, uint32_t r1, uint32_t numberOfElement )
	{
		a = r0;
		c = r1;
		n = numberOfElement;
		return gcd( a, n ) == 1;
	}
#endif
};

//DEVICE inline float fresnelSchlick( float cosTheta, float n1, float n2 )
//{
//	float r = ( n1 - n2 ) / ( n1 + n2 );
//	float R0 = r * r;
//	float k = 1.0f - cosTheta;
//	float kk = k * k;
//	return R0 + ( 1.0f - R0 ) * kk * kk * k;
//}
//DEVICE inline float3 reflect( float3 I, float3 N )
//{
//	return I - N * dot( N, I ) * 2.0f;
//}



