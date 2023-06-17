#include "pr.hpp"
#include <iostream>
#include <set>
#include <memory>
#include "voxelization.hpp"
#include "voxelMeshWriter.hpp"
#include "morton.hpp"

#include "IntersectorEmbree.hpp"

float scalbnf_bits( float x, int n )
{
	//uint32_t b = *(uint32_t*)&x;
	//b += 0x800000 * n;
	//return *(float*)&b;
	uint32_t b = *(uint32_t*)&x;
	int e = ( b >> 23 ) & 0xFF;
	e = glm::clamp( e + n, 0, 254 );
	b = ( b & ( ~0x7F800000 ) ) | (uint32_t)e << 23;
	return *(float*)&b;
}
float twopn( int n )
{
	uint32_t b = 0x3f800000 + 0x800000 * n;
	return *(float*)&b;
}
static bool experiment = true;



class SequencialHasher
{
public:
    void add(uint32_t xs, uint32_t resolution)
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
float maxElement(float a, float b, float c)
{
    return glm::max(glm::max(a, b), c);
}
float minElement(float a, float b, float c)
{
    return glm::min(glm::min(a, b), c);
}

inline void drawVoxels(const std::set<uint64_t>& mortonVoxels, const glm::vec3& origin, float dps, glm::u8vec3 color )
{
    using namespace pr;

    PrimBegin(PrimitiveMode::Lines);
    for (auto morton : mortonVoxels)
    {
        glm::uvec3 c;
        decodeMortonCode_PEXT(morton, &c.x, &c.y, &c.z);
        glm::vec3 p = origin + glm::vec3(c.x, c.y, c.z) * dps;

        uint32_t i0 = PrimVertex(p, color);
        uint32_t i1 = PrimVertex(p + glm::vec3(dps, 0, 0), color);
        uint32_t i2 = PrimVertex(p + glm::vec3(dps, 0, dps), color);
        uint32_t i3 = PrimVertex(p + glm::vec3(0, 0, dps), color);
        uint32_t i4 = PrimVertex(p + glm::vec3(0, dps, 0), color);
        uint32_t i5 = PrimVertex(p + glm::vec3(dps, dps, 0), color);
        uint32_t i6 = PrimVertex(p + glm::vec3(dps, dps, dps), color);
        uint32_t i7 = PrimVertex(p + glm::vec3(0, dps, dps), color);

        PrimIndex(i0); PrimIndex(i1);
        PrimIndex(i1); PrimIndex(i2);
        PrimIndex(i2); PrimIndex(i3);
        PrimIndex(i3); PrimIndex(i0);

        PrimIndex(i4); PrimIndex(i5);
        PrimIndex(i5); PrimIndex(i6);
        PrimIndex(i6); PrimIndex(i7);
        PrimIndex(i7); PrimIndex(i4);

        PrimIndex(i0); PrimIndex(i4);
        PrimIndex(i1); PrimIndex(i5);
        PrimIndex(i2); PrimIndex(i6);
        PrimIndex(i3); PrimIndex(i7);
    }
    PrimEnd();
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
        curTasks.push_back(task);
    }
    std::vector<OctreeTask> nextTasks;

    while (1 < wide)
    {
        std::vector<OctreeTask> sameParent;
        uint64_t parent = -1;

        auto emit = [&]()
        {
            // allocate
            uint32_t c = nodes->size();
            nodes->push_back(OctreeNode());
            (*nodes)[c].mask = 0;
            for (int i = 0; i < 8; i++)
            {
                (*nodes)[c].children[i] = -1;
            }

            // set child
            PR_ASSERT(sameParent.size() <= 8);
            for (int i = 0; i < sameParent.size(); i++)
            {
                uint32_t space = sameParent[i].morton & 0x7;
                (*nodes)[c].mask |= (1 << space) & 0xFF;
                (*nodes)[c].children[space] = sameParent[i].child;
            }

            OctreeTask nextTask;
            nextTask.morton = parent;
            nextTask.child = c;
            nextTasks.push_back(nextTask);

            parent = -1;
            sameParent.clear();
        };

        for (int i = 0; i < curTasks.size(); i++)
        {
            if (parent == -1)
            {
                sameParent.push_back(curTasks[i]);
                parent = curTasks[i].morton >> 3;
                continue;
            }

            if (parent == (curTasks[i].morton >> 3))
            {
                sameParent.push_back(curTasks[i]);
            }
            else
            {
                emit();
                sameParent.push_back(curTasks[i]);
                parent = curTasks[i].morton >> 3;
            }
        }

        if (sameParent.size())
        {
            emit();
        }

        curTasks.clear();
        std::swap(curTasks, nextTasks);

        wide /= 2;
    }
}
void drawAABBscaled(glm::vec3 lower, glm::vec3 upper, float scale, glm::u8vec3 color, float lineWidth = 1.0f)
{
    glm::vec3 c = (lower + upper) * 0.5f;
    glm::vec3 h = (upper - lower) * 0.5f;
    pr::DrawAABB(c - h * scale, c + h * scale, color, lineWidth);
}


glm::vec3 g_ro;
glm::vec3 g_rd;

void buildMaskList(uint32_t* list, float a, float b, float c)
{
    struct Order
    {
        uint32_t index0 : 2;
        uint32_t index1 : 2;
        uint32_t index2 : 2;
    };
    Order order = {};
    if (a < b)
    {
        order.index1++;
    }
    else
    {
        order.index0++;
    }
    if (b < c)
    {
        order.index2++;
    }
    else
    {
        order.index1++;
    }
    if (c < a)
    {
        order.index0++;
    }
    else
    {
        order.index2++;
    }
    uint32_t maskList[3];
    list[order.index0] = 1u;
    list[order.index1] = 2u;
    list[order.index2] = 4u;
}

void octreeTraverse_Hero(
    const std::vector<OctreeNode>& nodes, uint32_t nodeIndex,
    uint32_t vMask,
    float tx0, float ty0, float tz0,
    float tx1, float ty1, float tz1, float* t, int* nMajor, int depth = 0)
{
    // float tmin = glm::max(0.0f, maxElement(tx0, ty0, tz0));
    // float tmax = glm::min(*t, minElement(tx1, ty1, tz1));
    //float S_lmax = maxElement(tx0, ty0, tz0);
    //float S_umin = minElement(tx1, ty1, tz1);
    glm::vec3 t0{ tx0, ty0, tz0 };
    glm::vec3 t1{ tx1, ty1, tz1 };
    glm::vec3 tlower = glm::min(t0, t1);
    glm::vec3 tupper = glm::max(t0, t1);
    float S_lmax = maxElement(tlower.x, tlower.y, tlower.z);
    float S_umin = minElement(tupper.x, tupper.y, tupper.z);

    //glm::vec3 tmin = { tx0, ty0, tz0 };
    //glm::vec3 tmax = { tx1, ty1, tz1 };
    //if (vMask & 1u)
    //{
    //    std::swap(tmin.x, tmax.x);
    //}
    //if (vMask & 2u)
    //{
    //    std::swap(tmin.y, tmax.y);
    //}
    //if (vMask & 4u)
    //{
    //    std::swap(tmin.z, tmax.z);
    //}
    //float S_lmax = maxElement(tmin.x, tmin.y, tmin.z);
    //float S_umin = minElement(tmax.x, tmax.y, tmax.z);

    if( glm::min( S_umin, *t ) < glm::max( S_lmax, 0.0f ) )
        return;

    if (nodeIndex == -1)
    {
        if (S_lmax < *t)
        {
            *t = S_lmax;

            if (S_lmax == tlower.x)
            {
                *nMajor = 1;
            }
            else if (S_lmax == tlower.y)
            {
                *nMajor = 2;
            }
            else
            {
                *nMajor = 0;
            }
        }
        return;
    }

    float txM = 0.5f * ( tx0 + tx1 );
    float tyM = 0.5f * ( ty0 + ty1 );
    float tzM = 0.5f * ( tz0 + tz1 );

    uint32_t childMask =
        (txM < S_lmax ? 1u : 0u) |
        (tyM < S_lmax ? 2u : 0u) |
        (tzM < S_lmax ? 4u : 0u);
    uint32_t lastMask =
        (txM < S_umin ? 1u : 0u) |
        (tyM < S_umin ? 2u : 0u) |
        (tzM < S_umin ? 4u : 0u);

    uint32_t maskList[3];
    buildMaskList(maskList, txM, tyM, tzM );

    // glm::vec3 hsize = (upper - lower) * 0.5f;

    uint32_t currentChildMask = childMask;
    int i = 0;
    int bindex = 0;
    for (;;)
    {
        // process the child
        //glm::vec3 o =
        //{
        //    lower.x + ((currentChildMask ^ vMask) & 1u ? hsize.x : 0.0f),
        //    lower.y + ((currentChildMask ^ vMask) & 2u ? hsize.y : 0.0f),
        //    lower.z + ((currentChildMask ^ vMask) & 4u ? hsize.z : 0.0f),
        //};
        //drawAABBscaled(o, o + hsize, 0.93f, { 0,0, 255 }, 2);
        //pr::DrawText(o + hsize * 0.5f, std::to_string(bindex++));

        const OctreeNode& node = nodes[nodeIndex];
        uint32_t c = currentChildMask ^ vMask;
        if( node.mask & ( 0x1 << c ) )
        {
            float x0 = ( c & 1u ) ? txM : tx0;
            float x1 = ( c & 1u ) ? tx1 : txM;

            float y0 = ( c & 2u ) ? tyM : ty0;
            float y1 = ( c & 2u ) ? ty1 : tyM;

            float z0 = ( c & 4u ) ? tzM : tz0;
            float z1 = ( c & 4u ) ? tz1 : tzM;

            //if (vMask & 1u)
            //    std::swap(x0, x1);
            //if (vMask & 2u)
            //    std::swap(y0, y1);
            //if (vMask & 4u)
            //    std::swap(z0, z1);

            octreeTraverse_Hero( nodes, node.children[c], vMask, x0, y0, z0, x1, y1, z1, t, nMajor, depth + 1 );
        }

        if( currentChildMask == lastMask )
        {
            break;
        }

        //while( ( maskList[i] & currentChildMask ) != 0 )
        //{
        //    i++; // It means the flag is already set. Thus, skip OR operation
        //    if( 2 < i )
        //    {
        //        break;
        //    }
        //}
        //if (2 < i)
        //{
        //    break;
        //}

        //currentChildMask |= maskList[i];

        uint32_t nextChildMask = currentChildMask;
        do
        {
            nextChildMask = currentChildMask | maskList[i++];
        } while(nextChildMask == currentChildMask && i < 3 );
        if (nextChildMask == currentChildMask)
        {
            break;
        }
        currentChildMask = nextChildMask;
    }
}
void octreeTraverse_Hero(
    const std::vector<OctreeNode>& nodes, uint32_t nodeIndex,
    glm::vec3 ro,
    glm::vec3 one_over_rd,
    glm::vec3 lower,
    glm::vec3 upper,
    float* t, int* nMajor, int depth )
{
    glm::vec3 t0 = ( lower - ro ) * one_over_rd;
    glm::vec3 t1 = ( upper - ro ) * one_over_rd;
    //glm::vec3 tmin = glm::min( t0v, t1v );
    //glm::vec3 tmax = glm::max( t0v, t1v );
    //glm::vec3 tmin = t0v;
    //glm::vec3 tmax = t1v;
    //glm::vec3 tmid = ( tmin + tmax ) * 0.5f;

    //float S_lmax = maxElement( tmin.x, tmin.y, tmin.z );
    //float S_umin = minElement( tmax.x, tmax.y, tmax.z );

    glm::vec3 tmid = (t0 + t1) * 0.5f;

    glm::vec3 tlower = glm::min(t0, t1);
    glm::vec3 tupper = glm::max(t0, t1);
    float S_lmax = maxElement(tlower.x, tlower.y, tlower.z);
    float S_umin = minElement(tupper.x, tupper.y, tupper.z);

    if (glm::min(S_umin, *t) < glm::max(S_lmax, 0.0f))
        return;

    uint32_t vMask = 
        ( 0.0f < one_over_rd.x ? 0u : 1u ) |
        ( 0.0f < one_over_rd.y ? 0u : 2u ) |
        ( 0.0f < one_over_rd.z ? 0u : 4u );

    octreeTraverse_Hero(nodes, nodeIndex, vMask, t0.x, t0.y, t0.z, t1.x, t1.y, t1.z, t, nMajor, depth);

    //uint32_t childMask =
    //    ( tmid.x < S_lmax ? 1u : 0u ) |
    //    ( tmid.y < S_lmax ? 2u : 0u ) |
    //    ( tmid.z < S_lmax ? 4u : 0u );
    //uint32_t lastMask =
    //    ( tmid.x < S_umin ? 1u : 0u ) |
    //    ( tmid.y < S_umin ? 2u : 0u ) |
    //    ( tmid.z < S_umin ? 4u : 0u );

    //uint32_t maskList[3];
    //buildMaskList( maskList, tmid.x, tmid.y, tmid.z );

    //glm::vec3 hsize = (upper - lower) * 0.5f;

    //uint32_t currentChildMask = childMask;
    //int i = 0;
    //int bindex = 0;
    //for (;;)
    //{
    //    // process the child
    //    glm::vec3 o =
    //    {
    //        lower.x + ( (currentChildMask ^ vMask ) & 1u ? hsize.x : 0.0f),
    //        lower.y + ( (currentChildMask ^ vMask ) & 2u ? hsize.y : 0.0f),
    //        lower.z + ( (currentChildMask ^ vMask ) & 4u ? hsize.z : 0.0f),
    //    };
    //    drawAABBscaled(o, o + hsize, 0.93f, { 0,0, 255 }, 2);
    //    pr::DrawText(o + hsize * 0.5f, std::to_string(bindex++));

    //    if( currentChildMask == lastMask )
    //    {
    //        break;
    //    }

    //uint32_t nextChildMask = currentChildMask;
    //do
    //{
    //    nextChildMask = currentChildMask | maskList[i++];
    //} while (nextChildMask == currentChildMask && i < 3);
    //if (nextChildMask == currentChildMask)
    //{
    //    break;
    //}
    //currentChildMask = nextChildMask;
    //}
}

void octreeTraverse_EfficientParametric(
	const std::vector<OctreeNode>& nodes, uint32_t nodeIndex,
	uint32_t vMask,
	float tx0, float ty0, float tz0,
	float tx1, float ty1, float tz1, float* t, int* nMajor )
{
	float S_lmax = maxElement( tx0, ty0, tz0 );
	float S_umin = minElement( tx1, ty1, tz1 );

	if( glm::min( S_umin, *t ) < glm::max( S_lmax, 0.0f ) )
		return;

	if( nodeIndex == -1 )
	{
		if( S_lmax < *t )
		{
			*t = S_lmax;

			if( S_lmax == tx0 )
			{
				*nMajor = 1;
			}
			else if( S_lmax == ty0 )
			{
				*nMajor = 2;
			}
			else
			{
				*nMajor = 0;
			}
		}
		return;
	}

	float txM = 0.5f * ( tx0 + tx1 );
	float tyM = 0.5f * ( ty0 + ty1 );
	float tzM = 0.5f * ( tz0 + tz1 );

    uint32_t childMask =
		( txM < S_lmax ? 1u : 0u ) |
		( tyM < S_lmax ? 2u : 0u ) |
		( tzM < S_lmax ? 4u : 0u );

	for( ;; )
	{
		const OctreeNode& node = nodes[nodeIndex];
		float x1 = ( childMask & 1u ) ? tx1 : txM;
		float y1 = ( childMask & 2u ) ? ty1 : tyM;
		float z1 = ( childMask & 4u ) ? tz1 : tzM;
		if( node.mask & ( 0x1 << ( childMask ^ vMask ) ) )
		{
			float x0 = ( childMask & 1u ) ? txM : tx0;
			float y0 = ( childMask & 2u ) ? tyM : ty0;
			float z0 = ( childMask & 4u ) ? tzM : tz0;
			octreeTraverse_EfficientParametric( nodes, node.children[childMask ^ vMask], vMask, x0, y0, z0, x1, y1, z1, t, nMajor );
		}

		// find minimum( x1, y1, z1 ) for next hit
		uint32_t mv;
		if( x1 < y1 )
		{
			mv = x1 < z1 ? 1u : 4u;
		}
		else
		{
			mv = y1 < z1 ? 2u : 4u;
		}

		if( childMask & mv )
		{
			break;
		}
		childMask |= mv;
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

#if 0
    // Recursive ver
    octreeTraverse_EfficientParametric( nodes, nodeIndex, vMask, t0.x, t0.y, t0.z, t1.x, t1.y, t1.z, t, nMajor );
#else

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

#endif

}

void octreeTraverseNaive(
    const std::vector<OctreeNode>& nodes, uint32_t nodeIndex,
    glm::vec3 ro,
    glm::vec3 one_over_rd,
    glm::vec3 lower,
    glm::vec3 upper, 
    float* t, int* nMajor, int depth )
{
    glm::vec3 t0v = (lower - ro) * one_over_rd;
    glm::vec3 t1v = (upper - ro) * one_over_rd;
    glm::vec3 tmin = glm::min(t0v, t1v);
    glm::vec3 tmax = glm::max(t0v, t1v);

    // 0.0f, *t condition should be bad for normal
    float a = glm::max( 0.0f, maxElement( tmin.x, tmin.y, tmin.z ) );
    float b = glm::min( *t, minElement( tmax.x, tmax.y, tmax.z ) );

    if( b < a )
    {
        return;
    }
    if( nodeIndex == -1 )
    {
        if( a < *t )
        {
            *t = a;

            if (a == tmin.x)
            {
                *nMajor = 1;
            }
            else if (a == tmin.y)
            {
                *nMajor = 2;
            }
            else
            {
                *nMajor = 0;
            }
        }
        return;
    }
    const OctreeNode& node = nodes[nodeIndex];
    glm::vec3 mid = (lower + upper) * 0.5f;
    
    for( uint32_t i = 0; i < 8; i++ )
    {
        glm::vec3 l;
        glm::vec3 u;
        l.x = (i & 0x1) ? mid.x : lower.x;
        u.x = (i & 0x1) ? upper.x : mid.x;
        l.y = (i & 0x2) ? mid.y : lower.y;
        u.y = (i & 0x2) ? upper.y : mid.y;
        l.z = (i & 0x4) ? mid.z : lower.z;
        u.z = (i & 0x4) ? upper.z : mid.z;

        if (node.mask & ( 0x1u << i ) )
        {
            octreeTraverseNaive(nodes, node.children[i], ro, one_over_rd,
                l,
                u,
                t, nMajor, depth + 1
            );
        }
    }
}

int main() {
    using namespace pr;

    Config config;
    config.ScreenWidth = 1920;
    config.ScreenHeight = 1080;
    config.SwapInterval = 0;
    Initialize(config);

    Camera3D camera;
    camera.origin = { 4, 4, 4 };
    camera.lookat = { 0, 0, 0 };
    camera.zUp = false;

    double e = GetElapsedTime();

    const char* input = "bunny.obj";
    // const char* input = "Tri.obj";
    SetDataDir(ExecutableDir());
    std::string errorMsg;
    std::shared_ptr<FScene> scene = ReadWavefrontObj( GetDataPath(input), errorMsg );

    pr::ITexture* bgTexture = 0;

    while (pr::NextFrame() == false) {
        if (IsImGuiUsingMouse() == false) {
            UpdateCameraBlenderLike(&camera);
        }
        if (bgTexture) {
            ClearBackground(bgTexture);
        }
        else {
            ClearBackground(0.1f, 0.1f, 0.1f, 1);
        }

        BeginCamera(camera);

        PushGraphicState();

        DrawGrid(GridAxis::XZ, 1.0f, 10, { 128, 128, 128 });
        DrawXYZAxis(1.0f);

        static double voxel_time = 0.0f;
        static bool drawVoxelWire = true;
        static bool naiive = false;
        static bool embree = false;
#if 1
        static bool sixSeparating = true;
        static float dps = 0.1f;
        static glm::vec3 origin = { -2.0f, -2.0f, -2.0f };
        static int gridRes = 512;

        static glm::vec3 from = { -3, -3, -3 };
        static glm::vec3 to = { -0.415414095, 1.55378413, 1.55378413 };
        ManipulatePosition(camera, &from, 1);
        ManipulatePosition(camera, &to, 1);

        DrawText(from, "from");
        DrawText(to, "to");
        DrawLine(from, to, { 128 , 128 , 128 });

        static std::shared_ptr<IntersectorEmbree> embreeVoxel( new IntersectorEmbree() );
        static std::vector<OctreeNode> nodes;
        static bool buildVoxels = true;

        static glm::vec3 octree_lower;
        static glm::vec3 octree_upper;

        scene->visitPolyMesh([](std::shared_ptr<const FPolyMeshEntity> polymesh) {
            ColumnView<int32_t> faceCounts(polymesh->faceCounts());
            ColumnView<int32_t> indices(polymesh->faceIndices());
            ColumnView<glm::vec3> positions(polymesh->positions());

            // Geometry
            pr::PrimBegin(pr::PrimitiveMode::Lines);
            for (int i = 0; i < positions.count(); i++)
            {
                glm::vec3 p = positions[i];
                glm::ivec3 color = { 255,255,255 };
                pr::PrimVertex(p, { color });
            }
            int indexBase = 0;

            for (int i = 0; i < faceCounts.count(); i++)
            {
                int nVerts = faceCounts[i];
                for (int j = 0; j < nVerts; ++j)
                {
                    int i0 = indices[indexBase + j];
                    int i1 = indices[indexBase + (j + 1) % nVerts];
                    pr::PrimIndex(i0);
                    pr::PrimIndex(i1);
                }
                indexBase += nVerts;
            }
            pr::PrimEnd();

            // Assume Triangle

            if (buildVoxels == false)
            {
                return;
            }

            glm::vec3 lower = glm::vec3(FLT_MAX, FLT_MAX, FLT_MAX);
            glm::vec3 upper = glm::vec3(-FLT_MAX, -FLT_MAX, -FLT_MAX);

            for (int i = 0; i < faceCounts.count(); i++)
            {
                for (int j = 0; j < 3; ++j)
                {
                    int index = indices[i * 3 + j];
                    lower = glm::min(lower, positions[index]);
                    upper = glm::max(upper, positions[index]);
                }
            }

            // bounding box
            glm::vec3 size = upper - lower;
            float dps = glm::max(glm::max(size.x, size.y), size.z) / (float)gridRes;

            octree_lower = lower;
            octree_upper = lower + glm::vec3(dps, dps, dps) * (float)gridRes;

            DrawAABB(lower, lower + glm::vec3(dps, dps, dps) * (float)gridRes, { 255 ,0 ,0 });

            std::set<uint64_t> mortonVoxels;

            Stopwatch voxelsw;

            glm::vec3 origin = lower;

            for (int i = 0; i < faceCounts.count(); i++)
            {
                glm::vec3 v0 = positions[indices[i * 3]];
                glm::vec3 v1 = positions[indices[i * 3 + 1]];
                glm::vec3 v2 = positions[indices[i * 3 + 2]];

                VTContext context(v0, v1, v2, sixSeparating, origin, dps, gridRes);
                glm::ivec2 xrange = context.xRangeInclusive();
                for (int x = xrange.x; x <= xrange.y; x++)
                {
                    glm::ivec2 yrange = context.yRangeInclusive(x, dps);
                    for (int y = yrange.x; y <= yrange.y; y++)
                    {
                        glm::ivec2 zrange = context.zRangeInclusive(x, y, dps, sixSeparating);
                        for (int z = zrange.x; z <= zrange.y; z++)
                        {
                            glm::vec3 p = context.p(x, y, z, dps);
                            if (context.intersect(p))
                            {
                                glm::ivec3 c = context.i(x, y, z);
                                mortonVoxels.insert(encode2mortonCode_PDEP(c.x, c.y, c.z));
                            }
                        }
                    }
                }
            } // face

            voxel_time = voxelsw.elapsed();

            // Draw
            if( drawVoxelWire )
            {
                //for (auto morton : mortonVoxels)
                //{
                //    glm::uvec3 c;
                //    decodeMortonCode_PEXT(morton, &c.x, &c.y, &c.z);
                //    glm::vec3 p = origin + glm::vec3(c.x, c.y, c.z) * dps;
                //    DrawAABB(p, p + glm::vec3(dps, dps, dps), { 200 ,200 ,200 });
                //}
                drawVoxels(mortonVoxels, origin, dps, { 200 ,200 ,200 });
            }

            // voxel build
            buildOctree( &nodes, mortonVoxels, gridRes );

            embreeVoxel->build( mortonVoxels, origin, dps );
        });

        glm::vec3 ro = from;
        glm::vec3 rd = to - from;
        glm::vec3 one_over_rd = glm::vec3(1.0f) / rd;

        glm::vec3 t0 = (octree_lower - ro) * one_over_rd;
        glm::vec3 t1 = (octree_upper - ro) * one_over_rd;
        //glm::vec3 tmin = glm::min(t0, t1);
        //glm::vec3 tmax = glm::max(t0, t1);
        glm::vec3 tmin = t0;
        glm::vec3 tmax = t1;

        float a = maxElement(tmin.x, tmin.y, tmin.z);
        float b = minElement(tmax.x, tmax.y, tmax.z);

        //DrawSphere(ro + rd * a, 0.05f, { 255,0,0 });
        //DrawSphere(ro + rd * b, 0.05f, { 255,0,0 });

        g_ro = ro;
        g_rd = rd;

        //float rt0 = FLT_MAX;
        //int nMajor;
        //octreeTraverse_Hero( nodes, nodes.size() - 1,  tmin.x, tmin.y, tmin.z, tmax.x, tmax.y, tmax.z, &rt0, &nMajor );
        //DrawSphere(ro + rd * rt0, 0.05f, { 255,0,0 });

        float rt0 = FLT_MAX;
        int nMajor;
        // octreeTraverse_Hero( nodes, nodes.size() - 1, ro, one_over_rd, octree_lower, octree_upper, &rt0, &nMajor, 0 );
        octreeTraverse_EfficientParametric(nodes, nodes.size() - 1, ro, one_over_rd, octree_lower, octree_upper, &rt0, &nMajor );

        DrawSphere(ro + rd * rt0, 0.01f, { 255,0,0 });

        glm::vec3 hitN = unProjectPlane( { 0.0f, 0.0f }, project2plane_reminder( rd, nMajor ) < 0.0f ? 1.0f : -1.0f , nMajor );
        DrawArrow(ro + rd * rt0, ro + rd * rt0 + hitN * 0.1f, 0.01f, { 255,0,0 });

#if 1
        Image2DRGBA8 image;
        image.allocate(GetScreenWidth(), GetScreenHeight());

        CameraRayGenerator rayGenerator( GetCurrentViewMatrix(), GetCurrentProjMatrix(), image.width(), image.height() );


        //ParallelFor(image.height(), [&](int j) 
        for (int j = 0; j < image.height(); ++j)
        {
            for (int i = 0; i < image.width(); ++i)
            {
                glm::vec3 ro, rd;
                rayGenerator.shoot(&ro, &rd, i, j, 0.5f, 0.5f);
                glm::vec3 one_over_rd = glm::vec3(1.0f) / rd;

                float t = FLT_MAX;
                int nMajor;
                if (naiive)
                {
                    octreeTraverseNaive(nodes, nodes.size() - 1, ro, one_over_rd, octree_lower, octree_upper, &t, &nMajor, 0);
                }
                else if (embree)
                {
                    embreeVoxel->intersect( ro, rd, &t, &nMajor );
                }
                else
                {
					octreeTraverse_EfficientParametric( nodes, nodes.size() - 1, ro, one_over_rd, octree_lower, octree_upper, &t, &nMajor );
                }

                if( t != FLT_MAX ) {
                    glm::vec3 hitN = unProjectPlane( { 0.0f, 0.0f }, project2plane_reminder(rd, nMajor) < 0.0f ? 1.0f : -1.0f, nMajor);
                    glm::vec3 color = (hitN + glm::vec3(1.0f)) * 0.5f;
                    image(i, j) = { 255 * color.r, 255 * color.g, 255 * color.b, 255 };
                }
                else 
                {
                    image(i, j) = { 0, 0, 0, 255 };
                }
            }
        }
        //);
        if (bgTexture == nullptr) {
            bgTexture = CreateTexture();
        }
        bgTexture->upload(image);
#endif

#endif

#if 0
        float unit = 1.0f;
        static glm::vec3 v0 = { -unit, -unit - 0.3f, 0.0f };
        static glm::vec3 v1 = { unit , -unit, 0.0f };
        static glm::vec3 v2 = { -unit,  unit, 0.0f };

        ManipulatePosition(camera, &v0, 1);
        ManipulatePosition(camera, &v1, 1);
        ManipulatePosition(camera, &v2, 1);

        DrawText(v0, "v0");
        DrawText(v1, "v1");
        DrawText(v2, "v2");
        DrawLine(v0, v1, { 128 , 128 , 128 });
        DrawLine(v1, v2, { 128 , 128 , 128 });
        DrawLine(v2, v0, { 128 , 128 , 128 });

        static bool sixSeparating = true;
        static float dps = 0.1f;
        static glm::vec3 origin = { -2.0f, -2.0f, -2.0f };
        static int gridRes = 32;

        DrawText(origin, "origin");
        ManipulatePosition(camera, &origin, 1);
        DrawAABB(origin, origin + glm::vec3(dps, dps, dps) * (float)gridRes, { 255 ,0 ,0 });

        // VoxelTriangleIntersector intersector( v0, v1, v2, sixSeparating, dps );

        VTContext context(v0, v1, v2, sixSeparating, origin, dps, gridRes);
        glm::ivec2 xrange = context.xRangeInclusive();
        for (int x = xrange.x; x <= xrange.y; x++)
        {
            glm::ivec2 yrange = context.yRangeInclusive(x, dps);
            for (int y = yrange.x; y <= yrange.y; y++)
            {
                glm::ivec2 zrange = context.zRangeInclusive(x, y, dps, sixSeparating);
                for (int z = zrange.x; z <= zrange.y; z++)
                {
                    glm::vec3 p = context.p(x, y, z, dps);
                    if (context.intersect(p))
                    {
                        DrawAABB(p, p + glm::vec3(dps, dps, dps), { 200 ,200 ,200 });
                    }
                }
            }
        }
#endif
        PopGraphicState();
        EndCamera();

        BeginImGui();

        ImGui::SetNextWindowSize({ 500, 800 }, ImGuiCond_Once);
        ImGui::Begin("Panel");
        ImGui::Text("fps = %f", GetFrameRate());
        ImGui::Checkbox("drawVoxelWire", &drawVoxelWire);
        ImGui::Checkbox("buildVoxels", &buildVoxels);
        ImGui::Checkbox("naiive", &naiive);
        ImGui::Checkbox("embree", &embree);
        ImGui::Text("octree   = %d byte", (int)nodes.size() * sizeof(OctreeNode));
        ImGui::Text("embree = %lld byte", embreeVoxel->getMemoryConsumption() );
        
        //ImGui::InputInt("voxelX", &voxelX);
        //ImGui::InputInt("voxelY", &voxelY);
        //ImGui::InputInt("voxelZ", &voxelZ);

        //ImGui::Text("%llu", encode2mortonCode(voxelX, voxelY, voxelZ));
        //
        //ImGui::InputInt("NDepth", &NDepth);
        
#if 1
        ImGui::InputFloat("dps", &dps, 0.01f);
        ImGui::InputInt("gridRes", &gridRes);
        ImGui::Checkbox("sixSeparating", &sixSeparating);
        ImGui::Checkbox("experiment", &experiment);
        ImGui::Text("voxel: %f s", voxel_time);

        static int Resolution = 1024;
        ImGui::InputInt("Resolution", &Resolution);

        if (ImGui::Button("Voxelize As Mesh"))
        {
            const char* input = "bunny.obj";
            // const char* input = "Tri.obj";
            std::string errorMsg;
            std::shared_ptr<FScene> scene = ReadWavefrontObj(GetDataPath(input), errorMsg);
            std::shared_ptr<FPolyMeshEntity> polymesh = std::dynamic_pointer_cast<FPolyMeshEntity>(scene->entityAt(0));
            ColumnView<int32_t> faceCounts(polymesh->faceCounts());
            ColumnView<int32_t> indices(polymesh->faceIndices());
            ColumnView<glm::vec3> positions(polymesh->positions());

            // Assume Triangle
            glm::vec3 lower = glm::vec3(FLT_MAX, FLT_MAX, FLT_MAX);
            glm::vec3 upper = glm::vec3(-FLT_MAX, -FLT_MAX, -FLT_MAX);

            for (int i = 0; i < faceCounts.count(); i++)
            {
                for (int j = 0; j < 3; ++j)
                {
                    int index = indices[i * 3 + j];
                    lower = glm::min(lower, positions[index]);
                    upper = glm::max(upper, positions[index]);
                }
            }

            // bounding box
            glm::vec3 size = upper - lower;

            float dps = glm::max(glm::max(size.x, size.y), size.z) / (float)Resolution;

            // Voxelization
            std::set<uint64_t> voxels;
			VoxelMeshWriter writer;

            glm::vec3 origin = lower;

            for (int i = 0; i < faceCounts.count(); i++)
            {
                glm::vec3 v0 = positions[indices[i * 3]];
                glm::vec3 v1 = positions[indices[i * 3 + 1]];
                glm::vec3 v2 = positions[indices[i * 3 + 2]];

                VTContext context(v0, v1, v2, sixSeparating, origin, dps, Resolution);
                glm::ivec2 xrange = context.xRangeInclusive();
                for (int x = xrange.x; x <= xrange.y; x++)
                {
                    glm::ivec2 yrange = context.yRangeInclusive(x, dps);
                    for (int y = yrange.x; y <= yrange.y; y++)
                    {
                        glm::ivec2 zrange = context.zRangeInclusive(x, y, dps, sixSeparating);
                        for (int z = zrange.x; z <= zrange.y; z++)
                        {
                            glm::vec3 p = context.p(x, y, z, dps);
                            if (context.intersect(p))
                            {
                                glm::ivec3 c = context.i(x, y, z);
                                uint64_t m = encode2mortonCode_PDEP( c.x, c.y, c.z );
								writer.add( p, dps );
                                if( voxels.count( m ) == 0 )
                                {
                                    writer.add(p, dps);
                                }
                                else
                                {
									voxels.insert( m );
                                }
                            }
                        }
                    }
                }
            }
            writer.savePLY( GetDataPath( "vox.ply" ).c_str() );
        }
#endif
        
        ImGui::End();

        EndImGui();
    }

    pr::CleanUp();
}
