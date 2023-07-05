#pragma once

#include "vectorMath.hpp"

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

#define LP_OCCUPIED_BIT 0x80000000
#define LP_LOCK         0xFFFFFFFF
#define LP_VALUE_BIT    0x7FFFFFFF


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

				uint32_t nSkipped = 0;
				for( int i = 0; i < childIndex; i++ )
				{
					if( node.mask & ( 0x1 << i ) )
					{
						nSkipped += node.children[i] == -1 ? 1 : nodes[node.children[i]].numberOfVoxels;
					}
				}
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