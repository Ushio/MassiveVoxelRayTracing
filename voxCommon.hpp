#pragma once

#include "vectorMath.hpp"

#define ENABLE_GPU_DAG 1

#if defined( __CUDACC__ ) || defined( __HIPCC__ )
#ifndef DEVICE
#define DEVICE __device__
#endif
#else
#include <glm/glm.hpp>
#ifndef DEVICE
#define DEVICE
#endif
#endif

#if defined( CUDART_VERSION ) && CUDART_VERSION >= 9000
#define ITS 1
#endif

struct OctreeTask
{
	uint64_t morton;
	uint32_t child;
	uint32_t numberOfVoxels;

	DEVICE uint64_t getMortonParent() const { return morton >> 3; }
};

//DEVICE inline uint32_t hash( uint32_t x )
//{
//	x *= 0x9e3779b9u;
//	x ^= x >> 16;
//	return x;
//}

DEVICE inline uint32_t fmix32( uint32_t h )
{
	h ^= h >> 16;
	h *= 0x85ebca6b;
	h ^= h >> 13;
	h *= 0xc2b2ae35;
	h ^= h >> 16;

	return h;
}
DEVICE inline uint32_t rotl32( uint32_t x, char r )
{
	return ( x << r ) | ( x >> ( 32 - r ) );
}

struct MurmurHash32
{
	DEVICE MurmurHash32( uint32_t seed ) : h1( seed )
	{
	}
	DEVICE void combine( uint32_t k1 )
	{
		const uint32_t c1 = 0xcc9e2d51;
		const uint32_t c2 = 0x1b873593;

		k1 *= c1;
		k1 = rotl32( k1, 15 );
		k1 *= c2;

		h1 ^= k1;
		h1 = rotl32( h1, 13 );
		h1 = h1 * 5 + 0xe6546b64;

		len++;
	}
	DEVICE uint32_t getHash() const
	{
		return fmix32( h1 ^ ( len * 4 ) );
	}
	uint32_t h1 = 0;
	uint32_t len = 0;
};

DEVICE inline uint32_t hashCombine( uint32_t a, uint32_t b )
{
	MurmurHash32 hash( a );
	hash.combine( b );
	return hash.getHash();
}

DEVICE inline uint32_t hashCombine( uint32_t a, uint32_t b, uint32_t c )
{
	MurmurHash32 hash( a );
	hash.combine( b );
	hash.combine( c );
	return hash.getHash();
}
DEVICE inline uint32_t hashCombine( uint32_t a, uint32_t b, uint32_t c, uint32_t d )
{
	MurmurHash32 hash( a );
	hash.combine( b );
	hash.combine( c );
	hash.combine( d );
	return hash.getHash();
}
#if defined( __CUDACC__ ) || defined( __HIPCC__ )
// ...
#else
DEVICE inline int numberOfSortBitsMorton( uint32_t gridRes )
{
	unsigned long index;
	_BitScanForward( &index, gridRes );
	return index * 3;
}
#endif

struct VoxelAttirb
{
	uchar4 color;
	uchar4 emission;
};

struct EmissiveSurface
{
	float3 pivot;
	uchar4 emission;
};

struct OctreeNode
{
	uint8_t mask;
	uint32_t children[8];
	uint32_t nVoxelsPSum[8];

	DEVICE uint32_t getHash() const
	{
		MurmurHash32 h( 0 );
		h.combine( mask );
		for( int i = 0; i < 8; i++ )
			h.combine( children[i] );
		return h.getHash();
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

#define LP_OCCUPIED_BIT 0x80000000
#define LP_LOCK         0xFFFFFFFF
#define LP_VALUE_BIT    0x7FFFFFFF

#define SMALL_STACK 1
#if defined( SMALL_STACK )
struct alignas(16) StackElement
{
	uint32_t nodeIndex;
	float tx0;
	float ty0;
	float tz0;
	float scale;
	uint32_t childMask;
	uint32_t nVoxelSkipped;
	uint32_t _pad;
};
#else
struct StackElement
{
	uint32_t nodeIndex;
	float tx0;
	float ty0;
	float tz0;

	float S_lmax;
	float tx1;
	float ty1;
	float tz1;

	uint32_t childMask;
	uint32_t nVoxelSkipped;
};
#endif

DEVICE void octreeTraverse_EfficientParametric(
	const OctreeNode* nodes, uint32_t nodeIndex,
	StackElement *stack,
	float3 ro,
	float3 rd,
	const float3& lower,
	const float3& upper,
	float* t, int* nMajor, uint32_t* vIndex )
{
	float3 one_over_rd = float3{ 1.0f, 1.0f, 1.0f } / rd;

	uint32_t vMask = 0;
	if( one_over_rd.x < 0.0f )
	{
		vMask |= 1u;
		one_over_rd.x = -one_over_rd.x;
		ro.x = lower.x + upper.x - ro.x;
	}
	if( one_over_rd.y < 0.0f )
	{
		vMask |= 2u;
		one_over_rd.y = -one_over_rd.y;
		ro.y = lower.y + upper.y - ro.y;
	}
	if( one_over_rd.z < 0.0f )
	{
		vMask |= 4u;
		one_over_rd.z = -one_over_rd.z;
		ro.z = lower.z + upper.z - ro.z;
	}

	// const float kMinDir = 1.08420217249e-19f;
	// one_over_rd = glm::min( one_over_rd, glm::vec3( 1.0f / kMinDir ) );

	float3 bound = float3{ MAX_FLOAT, MAX_FLOAT, MAX_FLOAT } /
				   fmaxf(
					   fmaxf( fabs( lower - ro ), fabs( upper - ro ) ),
					   float3{ 1.0f, 1.0f, 1.0f } );
	one_over_rd = fminf( one_over_rd, bound );

	float3 t0 = ( lower - ro ) * one_over_rd;
	float3 t1 = ( upper - ro ) * one_over_rd;

	float S_lmaxTop = maxElement( t0.x, t0.y, t0.z );
	if( minElement( t1.x, t1.y, t1.z ) < S_lmaxTop ) // a case the box is totally behind of the ray is handled by the first condition of the loop
	{
		return;
	}
#if defined( SMALL_STACK )
	auto copyStackElement = []( StackElement& dst, const StackElement& src )
	{
		static_assert( sizeof( StackElement ) % 16 == 0, "" );
#if defined( __CUDACC__ )
		// faster on CUDA
		const float4* pSrc = (const float4*)&src;
		float4* pDst = (float4*)&dst;
		const int nCopy = sizeof( StackElement ) / sizeof( float4 );
		for( int i = 0; i < nCopy; i++ )
		{
			pDst[i] = pSrc[i];
		}
#else
		dst.nodeIndex = src.nodeIndex;
		dst.tx0 = src.tx0;
		dst.ty0 = src.ty0;
		dst.tz0 = src.tz0;

		dst.scale = src.scale;

		dst.childMask = src.childMask;
		dst.nVoxelSkipped = src.nVoxelSkipped;
#endif
	};
	int sp = 0;
	StackElement cur = { nodeIndex, t0.x, t0.y, t0.z, 1.0f, 0xFFFFFFFF, 0 };

	float3 dt = t1 - t0;

	for( ;; )
	{
	next:
		float tx1 = cur.tx0 + dt.x * cur.scale;
		float ty1 = cur.ty0 + dt.y * cur.scale;
		float tz1 = cur.tz0 + dt.z * cur.scale;

		// came here so that S_lmax < S_umin ; however, reject it when the box is totally behind. Otherwise, there are potential hits.
		if( minElement( tx1, ty1, tz1 ) < 0.0f )
		{
			goto pop;
		}

		float S_lmax = maxElement( cur.tx0, cur.ty0, cur.tz0 );

		if( cur.nodeIndex == -1 )
		{
			if( 0.0f < S_lmax ) // positive hit point only
			{
				*t = S_lmax; // S_lmax < *t is always true. max( a, 0 ) < min( b, t )  =>   a < t
				*nMajor =
					S_lmax == cur.tx0 ? 1 : ( S_lmax == cur.ty0 ? 2 : 0 );

				*vIndex = cur.nVoxelSkipped;
				// Since the traversal is in perfect order with respect to the ray direction, you can break it when you find a hit
				break;
			}
			goto pop;
		}

		float txM = 0.5f * ( cur.tx0 + tx1 );
		float tyM = 0.5f * ( cur.ty0 + ty1 );
		float tzM = 0.5f * ( cur.tz0 + tz1 );

		if( cur.childMask == 0xFFFFFFFF )
		{
			cur.childMask =
				( txM < S_lmax ? 1u : 0u ) |
				( tyM < S_lmax ? 2u : 0u ) |
				( tzM < S_lmax ? 4u : 0u );
		}

		const OctreeNode& node = nodes[cur.nodeIndex];

		float x1 = ( cur.childMask & 1u ) ? tx1 : txM;
		float y1 = ( cur.childMask & 2u ) ? ty1 : tyM;
		float z1 = ( cur.childMask & 4u ) ? tz1 : tzM;

		for( ;; )
		{
			// find minimum( x1, y1, z1 ) for next hit
			uint32_t mv =
				x1 < y1 ? ( x1 < z1 ? 1u : 4u ) : ( y1 < z1 ? 2u : 4u );

			bool hasNext = ( cur.childMask & mv ) == 0;
			uint32_t childIndex = cur.childMask ^ vMask;
			uint32_t currentChildMask = cur.childMask;
			cur.childMask |= mv;

			if( node.mask & ( 0x1 << childIndex ) )
			{
				if( hasNext )
				{
					copyStackElement( stack[sp++], cur );
				}
				cur.nodeIndex = node.children[childIndex];
				cur.tx0 = ( currentChildMask & 1u ) ? txM : cur.tx0;
				cur.ty0 = ( currentChildMask & 2u ) ? tyM : cur.ty0;
				cur.tz0 = ( currentChildMask & 4u ) ? tzM : cur.tz0;
				cur.scale *= 0.5f;

				cur.childMask = 0xFFFFFFFF;

				uint32_t nSkipped = node.nVoxelsPSum[childIndex];
				cur.nVoxelSkipped += nSkipped;

				goto next;
			}

			if( hasNext == false )
			{
				break;
			}
			switch( mv )
			{
			case 1:
				x1 = tx1;
				break;
			case 2:
				y1 = ty1;
				break;
			case 4:
				z1 = tz1;
				break;
			}
		}

	pop:
		if( sp )
		{
			copyStackElement( cur, stack[--sp] );
		}
		else
		{
			break;
		}
	}

#else

	auto copyStackElement = []( StackElement& dst, const StackElement& src )
	{
		dst.nodeIndex = src.nodeIndex;
		dst.tx0 = src.tx0;
		dst.ty0 = src.ty0;
		dst.tz0 = src.tz0;

		dst.S_lmax = src.S_lmax;
		dst.tx1 = src.tx1;
		dst.ty1 = src.ty1;
		dst.tz1 = src.tz1;

		dst.childMask = src.childMask;
		dst.nVoxelSkipped = src.nVoxelSkipped;
	};

	int sp = 0;
	StackElement cur = { nodeIndex, t0.x, t0.y, t0.z, S_lmaxTop, t1.x, t1.y, t1.z, 0xFFFFFFFF, 0 };

	for( ;; )
	{
	next:
		// came here so that S_lmax < S_umin ; however, reject it when the box is totally behind. Otherwise, there are potential hits.
		if( minElement( cur.tx1, cur.ty1, cur.tz1 ) < 0.0f )
		{
			goto pop;
		}

		if( cur.nodeIndex == -1 )
		{
			if( 0.0f < cur.S_lmax ) // positive hit point only
			{
				*t = cur.S_lmax; // S_lmax < *t is always true. max( a, 0 ) < min( b, t )  =>   a < t
				*nMajor =
					cur.S_lmax == cur.tx0 ? 1 : ( cur.S_lmax == cur.ty0 ? 2 : 0 );

				*vIndex = cur.nVoxelSkipped;
				// Since the traversal is in perfect order with respect to the ray direction, you can break it when you find a hit
				break;
			}
			goto pop;
		}

		float txM = 0.5f * ( cur.tx0 + cur.tx1 );
		float tyM = 0.5f * ( cur.ty0 + cur.ty1 );
		float tzM = 0.5f * ( cur.tz0 + cur.tz1 );

		if( cur.childMask == 0xFFFFFFFF )
		{
			cur.childMask =
				( txM < cur.S_lmax ? 1u : 0u ) |
				( tyM < cur.S_lmax ? 2u : 0u ) |
				( tzM < cur.S_lmax ? 4u : 0u );
		}

		const OctreeNode& node = nodes[cur.nodeIndex];

		float x1 = ( cur.childMask & 1u ) ? cur.tx1 : txM;
		float y1 = ( cur.childMask & 2u ) ? cur.ty1 : tyM;
		float z1 = ( cur.childMask & 4u ) ? cur.tz1 : tzM;

		for( ;; )
		{
			// find minimum( x1, y1, z1 ) for next hit
			uint32_t mv =
				x1 < y1 ? ( x1 < z1 ? 1u : 4u ) : ( y1 < z1 ? 2u : 4u );

			bool hasNext = ( cur.childMask & mv ) == 0;
			uint32_t childIndex = cur.childMask ^ vMask;
			uint32_t currentChildMask = cur.childMask;
			cur.childMask |= mv;

			if( node.mask & ( 0x1 << childIndex ) )
			{
				if( hasNext )
				{
					copyStackElement( stack[sp++], cur );
				}
				cur.nodeIndex = node.children[childIndex];
				cur.tx0 = ( currentChildMask & 1u ) ? txM : cur.tx0;
				cur.ty0 = ( currentChildMask & 2u ) ? tyM : cur.ty0;
				cur.tz0 = ( currentChildMask & 4u ) ? tzM : cur.tz0;
				cur.tx1 = x1;
				cur.ty1 = y1;
				cur.tz1 = z1;
				cur.S_lmax = maxElement( cur.tx0, cur.ty0, cur.tz0 );
				cur.childMask = 0xFFFFFFFF;

				uint32_t nSkipped = node.nVoxelsPSum[childIndex];
				cur.nVoxelSkipped += nSkipped;

				goto next;
			}

			if( hasNext == false )
			{
				break;
			}
			switch( mv )
			{
			case 1:
				x1 = cur.tx1;
				break;
			case 2:
				y1 = cur.ty1;
				break;
			case 4:
				z1 = cur.tz1;
				break;
			}
		}

	pop:
		if( sp )
		{
			copyStackElement( cur, stack[--sp] );
		}
		else
		{
			break;
		}
	}

#endif
}

template <class T>
DEVICE inline T getHitN( int major, T rd )
{
	switch( major )
	{
	case 0: // z
		return { 0.0f, 0.0f, 0.0f < rd.z ? -1.0f : 1.0f };
	case 1: // x
		return { 0.0f < rd.x ? -1.0f : 1.0f, 0.0f, 0.0f };
	case 2: // y
		return { 0.0f, 0.0f < rd.y ? -1.0f : 1.0f, 0.0f };
	}
	return { 0.0f, 0.0f, 0.0f };
}


template <class T>
DEVICE inline int bSearch( const T* xs, int n, T x )
{
	int i = 0;
	int j = n;

	while( i < j )
	{
		int m = ( i + j ) / 2;
		T value = xs[m];
		if( value == x )
		{
			return m;
		}
		else if( value < x )
		{
			i = m + 1;
		}
		else
		{
			j = m;
		}
	}
	return -1;
}