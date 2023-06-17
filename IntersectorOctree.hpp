#pragma once

#include <stdint.h>
#include <glm/glm.hpp>
#include <set>

#include "morton.hpp"


class SequencialHasher
{
public:
	void add( uint32_t xs, uint32_t resolution )
	{
		m_h += m_base * xs;
		m_base *= resolution;
	}
	uint32_t value() const { return m_h; }

private:
	uint32_t m_base = 1;
	uint32_t m_h = 0;
};

struct OctreeTask
{
	uint64_t morton;
	uint32_t child;
};
struct OctreeNode
{
	uint8_t mask;
	uint32_t children[8];
};
inline float maxElement( float a, float b, float c )
{
	return glm::max( glm::max( a, b ), c );
}
inline float minElement( float a, float b, float c )
{
	return glm::min( glm::min( a, b ), c );
}
inline void buildOctree( std::vector<OctreeNode>* nodes, const std::set<uint64_t>& mortonVoxels, int wide )
{
	nodes->clear();

	std::vector<OctreeTask> curTasks;
	for( auto m : mortonVoxels )
	{
		OctreeTask task;
		task.morton = m;
		task.child = -1;
		curTasks.push_back( task );
	}
	std::vector<OctreeTask> nextTasks;

	while( 1 < wide )
	{
		std::vector<OctreeTask> sameParent;
		uint64_t parent = -1;

		auto emit = [&]()
		{
			// allocate
			uint32_t c = nodes->size();
			nodes->push_back( OctreeNode() );
			( *nodes )[c].mask = 0;
			for( int i = 0; i < 8; i++ )
			{
				( *nodes )[c].children[i] = -1;
			}

			// set child
			PR_ASSERT( sameParent.size() <= 8 );
			for( int i = 0; i < sameParent.size(); i++ )
			{
				uint32_t space = sameParent[i].morton & 0x7;
				( *nodes )[c].mask |= ( 1 << space ) & 0xFF;
				( *nodes )[c].children[space] = sameParent[i].child;
			}

			OctreeTask nextTask;
			nextTask.morton = parent;
			nextTask.child = c;
			nextTasks.push_back( nextTask );

			parent = -1;
			sameParent.clear();
		};

		for( int i = 0; i < curTasks.size(); i++ )
		{
			if( parent == -1 )
			{
				sameParent.push_back( curTasks[i] );
				parent = curTasks[i].morton >> 3;
				continue;
			}

			if( parent == ( curTasks[i].morton >> 3 ) )
			{
				sameParent.push_back( curTasks[i] );
			}
			else
			{
				emit();
				sameParent.push_back( curTasks[i] );
				parent = curTasks[i].morton >> 3;
			}
		}

		if( sameParent.size() )
		{
			emit();
		}

		curTasks.clear();
		std::swap( curTasks, nextTasks );

		wide /= 2;
	}
}

void octreeTraverse_EfficientParametric(
	const std::vector<OctreeNode>& nodes, uint32_t nodeIndex,
	glm::vec3 ro,
	glm::vec3 one_over_rd,
	glm::vec3 lower,
	glm::vec3 upper,
	float* t, int* nMajor )
{
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

    glm::vec3 bound = glm::vec3( FLT_MAX ) / 
        glm::max( 
            glm::max( glm::abs( lower - ro ), glm::abs( upper - ro ) ), 
            glm::vec3( 1.0f ) 
        );
	one_over_rd = glm::min( one_over_rd, bound );

	glm::vec3 t0 = ( lower - ro ) * one_over_rd;
	glm::vec3 t1 = ( upper - ro ) * one_over_rd;

#if 1
    // lower number of stack ver

    float S_lmaxTop = maxElement( t0.x, t0.y, t0.z );
	if( minElement( t1.x, t1.y, t1.z ) < S_lmaxTop ) // a case the box is totally behind of the ray is handled by the first condition of the loop
	{
		return;
	}
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
	};
	auto copyStackElement = []( StackElement& dst, const StackElement& src ) {
		dst.nodeIndex = src.nodeIndex;
		dst.tx0 = src.tx0;
		dst.ty0 = src.ty0;
		dst.tz0 = src.tz0;

		dst.S_lmax = src.S_lmax;
		dst.tx1 = src.tx1;
		dst.ty1 = src.ty1;
		dst.tz1 = src.tz1;

		dst.childMask = src.childMask;
	};

    StackElement stack[32];
	int sp = 0;
	StackElement cur = { nodeIndex, t0.x, t0.y, t0.z, S_lmaxTop, t1.x, t1.y, t1.z, 0xFFFFFFFF };
	
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
					cur.S_lmax == cur.tx0 ? 1 : 
                        ( cur.S_lmax == cur.ty0 ? 2 : 
                            0 );

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
				x1 < y1 ? 
                    ( x1 < z1 ? 1u : 4u ) : 
                    ( y1 < z1 ? 2u : 4u );

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
#else
	// stackful ver
	struct StackElement
	{
		uint32_t nodeIndex;
		float tx0;
		float ty0;
		float tz0;
		float scale;
	};
	StackElement stack[512];
	int sp = 0;
	StackElement cur = { nodeIndex, t0.x, t0.y, t0.z, 1.0f };
	glm::vec3 dt = t1 - t0;
	for( ;; )
	{
		float S_lmax = maxElement( cur.tx0, cur.ty0, cur.tz0 );
		float tx1 = cur.tx0 + dt.x * cur.scale;
		float ty1 = cur.ty0 + dt.y * cur.scale;
		float tz1 = cur.tz0 + dt.z * cur.scale;
		
        // came here so that S_lmax < S_umin ; however, reject it when the box is totally behind
		if( minElement( tx1, ty1, tz1 ) < 0.0f )
		{
			goto pop;
		}

		if( cur.nodeIndex == -1 )
		{
			if( 0.0f < S_lmax ) // positive hit point only
			{
			    *t = S_lmax; // S_lmax < *t is always true. max( a, 0 ) < min( b, t )  =>   a < t

			    *nMajor =
					S_lmax == cur.tx0 ? 1 : 
                        ( S_lmax == cur.ty0 ? 2 : 
                            0 );

                // Since the traversal is in perfect order with respect to the ray direction, you can break it when you find a hit
			    break;
			}
			goto pop;
		}

		float txM = 0.5f * ( cur.tx0 + tx1 );
		float tyM = 0.5f * ( cur.ty0 + ty1 );
		float tzM = 0.5f * ( cur.tz0 + tz1 );

		uint32_t childMask =
			( txM < S_lmax ? 1u : 0u ) |
			( tyM < S_lmax ? 2u : 0u ) |
			( tzM < S_lmax ? 4u : 0u );

		uint32_t children = 0;
		int nChild = 0;

		const OctreeNode& node = nodes[cur.nodeIndex];
		for( ;; )
		{
			float x1 = ( childMask & 1u ) ? tx1 : txM;
			float y1 = ( childMask & 2u ) ? ty1 : tyM;
			float z1 = ( childMask & 4u ) ? tz1 : tzM;

			if( node.mask & ( 0x1 << ( childMask ^ vMask ) ) )
			{
				children = ( children << 3 ) | childMask;
				nChild++;
			}

			// find minimum( x1, y1, z1 ) for next hit
            uint32_t mv =
				x1 < y1 ? 
                    ( x1 < z1 ? 1u : 4u ) : 
                    ( y1 < z1 ? 2u : 4u );

			if( childMask & mv )
			{
				break;
			}
			childMask |= mv;
		}

        float hScale = cur.scale * 0.5f;
		for( int i = 0; i < nChild; i++ )
		{
			uint32_t child = ( children >> ( i * 3 ) ) & 0x7;
			float x0 = ( child & 1u ) ? txM : cur.tx0;
			float y0 = ( child & 2u ) ? tyM : cur.ty0;
			float z0 = ( child & 4u ) ? tzM : cur.tz0;

			if( i + 1 == nChild )
			{
				cur.nodeIndex = node.children[child ^ vMask];
				cur.tx0 = x0;
				cur.ty0 = y0;
				cur.tz0 = z0;
				cur.scale = hScale;
			}
			else
			{
				stack[sp].nodeIndex = node.children[child ^ vMask];
				stack[sp].tx0 = x0;
				stack[sp].ty0 = y0;
				stack[sp].tz0 = z0;
				stack[sp].scale = hScale;
				sp++;
			}
		}

		if( nChild )
		{
			continue;
		}
	pop:
		if( sp )
		{
			cur = stack[--sp];
		}
		else
		{
			break;
		}
	}
#endif
}
class IntersectorOctree
{
public:
	IntersectorOctree()
	{
	}

	void build( const std::set<uint64_t>& mortonVoxels, const glm::vec3& origin, float dps, int gridRes )
	{
		m_origin = origin;
		m_dps = dps;
		m_gridRes = gridRes;
		buildOctree( &m_nodes, mortonVoxels, gridRes );
	}
	void intersect( glm::vec3 ro, glm::vec3 rd, float* t, int* nMajor )
	{
		glm::vec3 upper = m_origin + glm::vec3( m_dps, m_dps, m_dps ) * (float)m_gridRes;
		glm::vec3 one_over_rd = glm::vec3( 1.0f ) / rd;
		octreeTraverse_EfficientParametric( m_nodes, m_nodes.size() - 1, ro, one_over_rd, m_origin, upper, t, nMajor );
	}

	uint64_t getMemoryConsumption()
	{
		return m_nodes.size() * sizeof( OctreeNode );
	}
	glm::vec3 m_origin = glm::vec3( 0.0f );
	float m_dps = 0.0f;
	int m_gridRes = 0;
	std::vector<OctreeNode> m_nodes;
};