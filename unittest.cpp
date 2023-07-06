#include "utest.h"
// https://github.com/sheredom/utest.h

#include "pr.hpp"
#include "morton.hpp"
#include "voxCommon.hpp"
#include "MurmurHash3.h"

UTEST_MAIN();

UTEST( MurmurHash3, compatibility )
{
	using namespace pr;

	Xoshiro128StarStar random;
	for( int i = 0; i < 100000000; i++ )
	{
		uint32_t xs[16];
		uint32_t n = random.uniformi() % 16;
		for (int j = 0; j < n; j++)
		{
			xs[j] = (uint32_t)random.uniformi();
		}
		uint32_t s = (uint32_t)random.uniformi();
		uint32_t h0;
		MurmurHash3_x86_32( xs, n * 4, s, &h0 );

		
		MurmurHash32 hasher( s );
		for (int j = 0; j < n; j++)
		{
			hasher.combine( xs[j] );
		}
		uint32_t h1 = hasher.getHash();
		ASSERT_EQ( h0, h1 );
	}
}
UTEST( morton, benchmark )
{
	using namespace pr;
	// perf
	{
		uint64_t k = 0;
		Stopwatch sw;
		Xoshiro128StarStar random;
		for( int i = 0; i < 100000000; i++ )
		{
			uint32_t x = random.uniformi() & 0x1FFFFF; // 21 bits
			uint32_t y = random.uniformi() & 0x1FFFFF; // 21 bits
			uint32_t z = random.uniformi() & 0x1FFFFF; // 21 bits

			uint64_t m0 = encode2mortonCode_Naive( x, y, z );
			k += m0;
		}
		printf( "%f s encode2mortonCode, %lld\n", sw.elapsed(), k );
	}
	{
		uint64_t k = 0;
		Stopwatch sw;
		Xoshiro128StarStar random;
		for( int i = 0; i < 100000000; i++ )
		{
			uint32_t x = random.uniformi() & 0x1FFFFF; // 21 bits
			uint32_t y = random.uniformi() & 0x1FFFFF; // 21 bits
			uint32_t z = random.uniformi() & 0x1FFFFF; // 21 bits

			uint64_t m0 = encode2mortonCode_magicbits( x, y, z );
			k += m0;
		}
		printf( "%f s encode2mortonCode_magicbits, %lld\n", sw.elapsed(), k );
	}
	{
		uint64_t k = 0;
		Stopwatch sw;
		Xoshiro128StarStar random;
		for( int i = 0; i < 100000000; i++ )
		{
			uint32_t x = random.uniformi() & 0x1FFFFF; // 21 bits
			uint32_t y = random.uniformi() & 0x1FFFFF; // 21 bits
			uint32_t z = random.uniformi() & 0x1FFFFF; // 21 bits

			uint64_t m0 = encode2mortonCode_PDEP( x, y, z );
			k += m0;
		}
		printf( "%f s encode2mortonCode_PDEP, %lld\n", sw.elapsed(), k );
	}
}
UTEST( morton, encodedecode )
{
	using namespace pr;
	Xoshiro128StarStar random;
	for( int i = 0; i < 100000000; i++ )
	{
		uint32_t x = random.uniformi() & 0x1FFFFF; // 21 bits
		uint32_t y = random.uniformi() & 0x1FFFFF; // 21 bits
		uint32_t z = random.uniformi() & 0x1FFFFF; // 21 bits

		uint64_t m0 = encode2mortonCode_Naive( x, y, z );
		uint64_t m1 = encode2mortonCode_PDEP( x, y, z );
		uint64_t m2 = encode2mortonCode_magicbits( x, y, z );
		ASSERT_EQ( m0, m1 );
		ASSERT_EQ( m0, m2 );
			
		uint32_t dx, dy, dz;
		decodeMortonCode_Naive( m0, &dx, &dy, &dz );
		ASSERT_EQ( x, dx );
		ASSERT_EQ( y, dy );
		ASSERT_EQ( z, dz );

		decodeMortonCode_PEXT( m0, &dx, &dy, &dz );
		ASSERT_EQ( x, dx );
		ASSERT_EQ( y, dy );
		ASSERT_EQ( z, dz );
	}
}