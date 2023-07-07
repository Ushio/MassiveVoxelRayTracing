#pragma once

#include "vectorMath.hpp"

#if defined( __CUDACC__ ) || defined( __HIPCC__ )

#else
#include <glm/glm.hpp>
#include "Orochi/Orochi.h"
#endif

class CameraPinhole
{
public:
#if !defined( __CUDACC__ ) && !defined( __HIPCC__ )
	void initFromPerspective( glm::mat4 viewMatrix, glm::mat4 projMatrix )
	{
		glm::mat3 vT = glm::transpose( glm::mat3( viewMatrix ) );
		m_front = { -vT[2].x, -vT[2].y, -vT[2].z };
		m_up = { vT[1].x, vT[1].y, vT[1].z };
		m_right = { vT[0].x, vT[0].y, vT[0].z };

		glm::vec3 m = vT * glm::vec3( viewMatrix[3] );
		m_o = { -m.x, -m.y, -m.z };

		m_tanHthetaY = 1.0f / projMatrix[1][1];
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
	float3 m_o;
	float3 m_front;
	float3 m_up;
	float3 m_right;
	float m_tanHthetaY;
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

// forward: +x, up: +y
DEVICE inline float2 getSpherical( float3 n )
{
	float phi = atan2f( n.z, n.x ) + PI;
	float theta = atan2f( INTRIN_SQRT( n.x * n.x + n.z * n.z ), n.y );
	return { phi / ( PI * 2.0f ), theta / PI };
}

struct HDRI
{
#if !defined( __CUDACC__ ) && !defined( __HIPCC__ )
	void load( float* ptr, int width, int height, oroStream stream )
	{
		m_width = width;
		m_height = height;
		if( m_pixels )
		{
			oroFree( (oroDeviceptr)m_pixels );
		}
		uint64_t bytes = sizeof( float4 ) * 4 * m_width * m_height;
		oroMalloc( (oroDeviceptr*)&m_pixels, bytes );
		oroMemcpyHtoDAsync( (oroDeviceptr)m_pixels, ptr, bytes, stream );
		oroStreamSynchronize( stream );
	}
	void cleanUp()
	{
		if( m_pixels )
		{
			oroFree( (oroDeviceptr)m_pixels );
			m_pixels = 0;
		}
	}
#else
	DEVICE float4 sampleNearest( float3 direction ) const
	{
		float2 uv = getSpherical( direction );
		int x = (int)ss_clamp( uv.x * m_width, 0.0f, (float)( m_width - 1.0f ) );
		int y = (int)ss_clamp( uv.x * m_height, 0.0f, (float)( m_width - 1.0f ) );
		uint64_t index = (uint64_t)y * m_width + x;
		return m_pixels[index];
	}
#endif

	float4* m_pixels;
	int m_width;
	int m_height;
};
