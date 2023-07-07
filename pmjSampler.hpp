#pragma once

#include "renderCommon.hpp"

#if !defined( __CUDACC__ ) && !defined( __HIPCC__ )
#include <functional>
#include <iostream>
#include <memory>
#include <vector>

// simpler vesion of
// https://github.com/Andrew-Helmer/stochastic-generation
// https://github.com/Andrew-Helmer/stochastic-generation/blob/main/LICENSE
inline void GetPMJ02Samples(
	const int num_samples,
	float* samples,
	std::function<float( void )> uniformFloat )
{
	const int nd = 2;
	static constexpr uint32_t pmj02_xors[2][32] = {
		{ 0x0, 0x0, 0x2, 0x6, 0x6, 0xe, 0x36, 0x4e, 0x16, 0x2e, 0x276, 0x6ce, 0x716, 0xc2e, 0x3076, 0x40ce, 0x116, 0x22e, 0x20676, 0x60ece, 0x61716, 0xe2c2e, 0x367076, 0x4ec0ce, 0x170116, 0x2c022e, 0x2700676, 0x6c00ece, 0x7001716, 0xc002c2e, 0x30007076, 0x4000c0ce },
		{ 0x0, 0x1, 0x3, 0x3, 0x7, 0x1b, 0x27, 0xb, 0x17, 0x13b, 0x367, 0x38b, 0x617, 0x183b, 0x2067, 0x8b, 0x117, 0x1033b, 0x30767, 0x30b8b, 0x71617, 0x1b383b, 0x276067, 0xb808b, 0x160117, 0x138033b, 0x3600767, 0x3800b8b, 0x6001617, 0x1800383b, 0x20006067, 0x808b } };
	auto GetPMJ02Point = [](
							 int x_stratum, int y_stratum,
							 float i_strata,
							 float xi0, float xi1,
							 float* sample )
	{
		sample[0] = ( xi0 + x_stratum ) * i_strata;
		sample[1] = ( xi1 + y_stratum ) * i_strata;
	};

	// Generate first sample randomly.
	for( int d = 0; d < 2; d++ )
	{
		samples[d] = uniformFloat();
	}

	for( int log_n = 0; ( 1 << log_n ) < num_samples; log_n++ )
	{
		int prev_len = 1 << log_n;
		int n_strata = prev_len * 2;
		float i_strata = 1.0f / n_strata;
		for( int i = 0; i < prev_len && ( prev_len + i ) < num_samples; i++ )
		{
			const int prev_x_idx = i ^ pmj02_xors[0][log_n];
			const int prev_x_stratum = samples[prev_x_idx * 2] * n_strata;
			const int x_stratum = prev_x_stratum ^ 1;

			const int prev_y_idx = i ^ pmj02_xors[1][log_n];
			const int prev_y_stratum = samples[prev_y_idx * 2 + 1] * n_strata;
			const int y_stratum = prev_y_stratum ^ 1;

			float* sample = &( samples[( prev_len + i ) * 2] );
			GetPMJ02Point( x_stratum, y_stratum, i_strata, uniformFloat(), uniformFloat(), sample );
		}
	}
}

#endif

DEVICE inline uint32_t laine_karras_permutation( uint32_t x, uint32_t seed )
{
	x += seed;
	x ^= x * 0x6c50b47cu;
	x ^= x * 0xb82f1e52u;
	x ^= x * 0xc7afe638u;
	x ^= x * 0x8d22f6e6u;
	return x;
}

// https://graphics.stanford.edu/~seander/bithacks.html#BitReverseTable
DEVICE inline uint32_t reverseBits( uint32_t v )
{
	// swap odd and even bits
	v = ( ( v >> 1 ) & 0x55555555 ) | ( ( v & 0x55555555 ) << 1 );
	// swap consecutive pairs
	v = ( ( v >> 2 ) & 0x33333333 ) | ( ( v & 0x33333333 ) << 2 );
	// swap nibbles ...
	v = ( ( v >> 4 ) & 0x0F0F0F0F ) | ( ( v & 0x0F0F0F0F ) << 4 );
	// swap bytes
	v = ( ( v >> 8 ) & 0x00FF00FF ) | ( ( v & 0x00FF00FF ) << 8 );
	// swap 2-byte long pairs
	v = ( v >> 16 ) | ( v << 16 );
	return v;
}

DEVICE inline uint32_t nested_uniform_scramble( uint32_t x, uint32_t seed )
{
	x = reverseBits( x );
	x = laine_karras_permutation( x, seed );
	x = reverseBits( x );
	return x;
}

DEVICE inline float scramble_f32( float x, uint32_t seed )
{
	x += 1.0f; // [1.0f, 2.0f)
	uint32_t bits = *reinterpret_cast<uint32_t*>( &x );
	uint32_t scrambled = 0x3f800000 | ( nested_uniform_scramble( bits & 0x7FFFFF, seed ) & 0x7FFFFF );
	return *reinterpret_cast<float*>( &scrambled ) - 1.0f;
}

class PMJSampler
{
public:
	enum
	{
		LENGTH = 4096,
		N_SEQUENCE = 128,
	};

#if !defined( __CUDACC__ ) && !defined( __HIPCC__ )
	void setup( bool isGPU, oroStream stream )
	{
		PCG32 rng;
		rng.setup( 0, 2525 );
		std::vector<float> samples( 2 * LENGTH * N_SEQUENCE );
		for( int i = 0; i < N_SEQUENCE; i++ )
		{
			float* p = samples.data() + 2 * LENGTH * i;
			GetPMJ02Samples( LENGTH, p, [&rng](){ return uniformf( rng.nextU32() ); } );
		}

		if( isGPU )
		{
			if( m_samples )
			{
				oroFree( (oroDeviceptr)m_samples );
			}
			oroMalloc( (oroDeviceptr*)&m_samples, sizeof( float ) * 2 * LENGTH * N_SEQUENCE );
			oroMemcpyHtoDAsync( (oroDeviceptr)m_samples, samples.data(), sizeof( float ) * 2 * LENGTH * N_SEQUENCE, stream );
			oroStreamSynchronize( stream );
		}
		else
		{
			if( m_samples )
			{
				free( m_samples );
			}
			m_samples = (float*)malloc( sizeof( float ) * 2 * LENGTH * N_SEQUENCE );
			memcpy( m_samples, samples.data(), sizeof( float ) * 2 * LENGTH * N_SEQUENCE );
		}
	}
	void cleanUp( bool isGPU )
	{
		if( isGPU )
		{
			if( m_samples )
			{
				oroFree( (oroDeviceptr)m_samples );
				m_samples = 0;
			}
		}
	}
#endif

	DEVICE float2 sample2d( uint32_t sampleIdx, uint32_t dimension, uint32_t stream )
	{
		// Shuffling
		sampleIdx = nested_uniform_scramble( sampleIdx, hashCombine( stream, dimension, 31082745 ) ) & ( LENGTH - 1 );
		dimension = nested_uniform_scramble( dimension, hashCombine( stream, 54761983 ) ) & ( N_SEQUENCE - 1 );

		uint32_t head = dimension * 2 * LENGTH;
		float x = m_samples[head + sampleIdx * 2 + 0];
		float y = m_samples[head + sampleIdx * 2 + 1];

		// Scrambling
		x = scramble_f32( x, hashCombine( stream, dimension, 83927105 ) );
		y = scramble_f32( y, hashCombine( stream, dimension, 12654890 ) );
		
		return { x, y };
	}
private:
	float* m_samples = 0;
};