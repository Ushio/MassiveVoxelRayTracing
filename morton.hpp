#pragma once

#include <immintrin.h>

inline uint64_t encode2mortonCode_Naive( uint32_t x, uint32_t y, uint32_t z )
{
	uint64_t code = 0;
	for( uint64_t i = 0; i < 64 / 3; ++i )
	{
		code |=
			( (uint64_t)( x & ( 1u << i ) ) << ( 2 * i + 0 ) ) |
			( (uint64_t)( y & ( 1u << i ) ) << ( 2 * i + 1 ) ) |
			( (uint64_t)( z & ( 1u << i ) ) << ( 2 * i + 2 ) );
	}
	return code;
}

inline void decodeMortonCode_Naive( uint64_t morton, uint32_t* x, uint32_t* y, uint32_t* z )
{
	uint32_t ox = 0;
	uint32_t oy = 0;
	uint32_t oz = 0;
	for( uint64_t i = 0; i < 64 / 3; ++i )
	{
		uint64_t a = morton & 0x1;
		uint64_t b = ( morton & 0x2 ) >> 1;
		uint64_t c = ( morton & 0x4 ) >> 2;
		morton = morton >> 3;
		ox |= a << i;
		oy |= b << i;
		oz |= c << i;
	}
	*x = ox;
	*y = oy;
	*z = oz;
}

/*
* https://www.chessprogramming.org/BMI2
SRC1   ┌───┬───┬───┬───┬───┐    ┌───┬───┬───┬───┬───┬───┬───┬───┐
	   │S63│S62│S61│S60│S59│....│ S7│ S6│ S5│ S4│ S3│ S2│ S1│ S0│
	   └───┴───┴───┴───┴───┘    └───┴───┴───┴───┴───┴───┴───┴───┘

SRC2   ┌───┬───┬───┬───┬───┐    ┌───┬───┬───┬───┬───┬───┬───┬───┐
(mask) │ 0 │ 0 │ 0 │ 1 │ 0 │0...│ 1 │ 0 │ 1 │ 0 │ 0 │ 1 │ 0 │ 0 │  (f.i. 4 bits set)
	   └───┴───┴───┴───┴───┘    └───┴───┴───┴───┴───┴───┴───┴───┘

DEST   ┌───┬───┬───┬───┬───┐    ┌───┬───┬───┬───┬───┬───┬───┬───┐
	   │ 0 │ 0 │ 0 │ S3│ 0 │0...│ S2│ 0 │ S1│ 0 │ 0 │ S0│ 0 │ 0 │
	   └───┴───┴───┴───┴───┘    └───┴───┴───┴───┴───┴───┴───┴───┘
*/

inline uint64_t encode2mortonCode_PDEP( uint32_t x, uint32_t y, uint32_t z )
{
	uint64_t code =
		_pdep_u64( x & 0x1FFFFF, 0x1249249249249249LLU ) |
		_pdep_u64( y & 0x1FFFFF, 0x1249249249249249LLU << 1 ) |
		_pdep_u64( z & 0x1FFFFF, 0x1249249249249249LLU << 2 );
	return code;
}

/*
* https://www.chessprogramming.org/BMI2#PEXT
SRC1   ┌───┬───┬───┬───┬───┐    ┌───┬───┬───┬───┬───┬───┬───┬───┐
	   │S63│S62│S61│S60│S59│....│ S7│ S6│ S5│ S4│ S3│ S2│ S1│ S0│
	   └───┴───┴───┴───┴───┘    └───┴───┴───┴───┴───┴───┴───┴───┘

SRC2   ┌───┬───┬───┬───┬───┐    ┌───┬───┬───┬───┬───┬───┬───┬───┐
(mask) │ 0 │ 0 │ 0 │ 1 │ 0 │0...│ 1 │ 0 │ 1 │ 0 │ 0 │ 1 │ 0 │ 0 │  (f.i. 4 bits set)
	   └───┴───┴───┴───┴───┘    └───┴───┴───┴───┴───┴───┴───┴───┘

DEST   ┌───┬───┬───┬───┬───┐    ┌───┬───┬───┬───┬───┬───┬───┬───┐
	   │ 0 │ 0 │ 0 │ 0 │ 0 │0...│ 0 │ 0 │ 0 │ 0 │S60│ S7│ S5│ S2│
	   └───┴───┴───┴───┴───┘    └───┴───┴───┴───┴───┴───┴───┴───┘
*/
inline void decodeMortonCode_PEXT( uint64_t morton, uint32_t* x, uint32_t* y, uint32_t* z )
{
	*x = static_cast<uint32_t>( _pext_u64( morton, 0x1249249249249249LLU ) );
	*y = static_cast<uint32_t>( _pext_u64( morton, 0x1249249249249249LLU << 1 ) );
	*z = static_cast<uint32_t>( _pext_u64( morton, 0x1249249249249249LLU << 2 ) );
}

// method to seperate bits from a given integer 3 positions apart
inline uint64_t splitBy3( uint32_t a )
{
	uint64_t x = a & 0x1FFFFF;
	x = ( x | x << 32 ) & 0x1f00000000ffff;	 // shift left 32 bits, OR with self, and 00011111000000000000000000000000000000001111111111111111
	x = ( x | x << 16 ) & 0x1f0000ff0000ff;	 // shift left 32 bits, OR with self, and 00011111000000000000000011111111000000000000000011111111
	x = ( x | x << 8 ) & 0x100f00f00f00f00f; // shift left 32 bits, OR with self, and 0001000000001111000000001111000000001111000000001111000000000000
	x = ( x | x << 4 ) & 0x10c30c30c30c30c3; // shift left 32 bits, OR with self, and 0001000011000011000011000011000011000011000011000011000100000000
	x = ( x | x << 2 ) & 0x1249249249249249;
	return x;
}
inline uint64_t encode2mortonCode_magicbits( uint32_t x, uint32_t y, uint32_t z )
{
	uint64_t answer = 0;
	answer |= splitBy3( x ) | splitBy3( y ) << 1 | splitBy3( z ) << 2;
	return answer;
}