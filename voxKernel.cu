#include "vectorMath.hpp"
#include "voxelization.hpp"

#include "voxCommon.hpp"

// method to seperate bits from a given integer 3 positions apart
__device__ inline uint64_t splitBy3( uint32_t a )
{
	uint64_t x = a & 0x1FFFFF;
	x = ( x | x << 32 ) & 0x1f00000000ffff;	 // shift left 32 bits, OR with self, and 00011111000000000000000000000000000000001111111111111111
	x = ( x | x << 16 ) & 0x1f0000ff0000ff;	 // shift left 32 bits, OR with self, and 00011111000000000000000011111111000000000000000011111111
	x = ( x | x << 8 ) & 0x100f00f00f00f00f; // shift left 32 bits, OR with self, and 0001000000001111000000001111000000001111000000001111000000000000
	x = ( x | x << 4 ) & 0x10c30c30c30c30c3; // shift left 32 bits, OR with self, and 0001000011000011000011000011000011000011000011000011000100000000
	x = ( x | x << 2 ) & 0x1249249249249249;
	return x;
}
__device__ inline uint64_t encode2mortonCode_magicbits( uint32_t x, uint32_t y, uint32_t z )
{
	uint64_t answer = 0;
	answer |= splitBy3( x ) | splitBy3( y ) << 1 | splitBy3( z ) << 2;
	return answer;
}

template <int NElement, int NThread, class T>
__device__ void clearShared( T* sMem, T value )
{
	for( int i = 0; i < NElement; i += NThread )
	{
		if( i < NElement )
		{
			sMem[i + threadIdx.x] = value;
		}
	}
}

extern "C" __global__ void voxCount( const float3 *vertices, const float3 *vcolors, uint32_t nTriangles, uint32_t* counter, float3 origin, float dps, uint32_t gridRes )
{
    uint32_t iTri = blockIdx.x * blockDim.x + threadIdx.x;

    if( iTri < nTriangles )
    {
        float3 v0 = vertices[iTri * 3];
        float3 v1 = vertices[iTri * 3 + 1];
        float3 v2 = vertices[iTri * 3 + 2];

        bool sixSeparating = true;
        VTContext context( v0, v1, v2, sixSeparating, { origin.x, origin.y, origin.z }, dps, gridRes );
        int2 xrange = context.xRangeInclusive();
        uint32_t nVoxels = 0;
        for( int x = xrange.x; x <= xrange.y; x++ )
        {
            int2 yrange = context.yRangeInclusive( x, dps );
            for( int y = yrange.x; y <= yrange.y; y++ )
            {
                int2 zrange = context.zRangeInclusive( x, y, dps, sixSeparating );
                for( int z = zrange.x; z <= zrange.y; z++ )
                {
                    float3 p = context.p( x, y, z, dps );
                    if( context.intersect( p ) )
                    {
                        nVoxels++;
                    }
                }
            }
        }
        atomicAdd( counter, nVoxels );
    }
}
extern "C" __global__ void voxelize( const float3 *vertices, const float3 *vcolors, uint32_t nTriangles, uint32_t* counter, float3 origin, float dps, uint32_t gridRes, uint64_t* mortonVoxels, uchar4* voxelColors )
{
    uint32_t iTri = blockIdx.x * blockDim.x + threadIdx.x;

    if( iTri < nTriangles )
    {
        float3 v0 = vertices[iTri * 3];
        float3 v1 = vertices[iTri * 3 + 1];
        float3 v2 = vertices[iTri * 3 + 2];

        float3 c0 = vcolors[iTri * 3];
        float3 c1 = vcolors[iTri * 3 + 1];
        float3 c2 = vcolors[iTri * 3 + 2];

        bool sixSeparating = true;
        VTContext context( v0, v1, v2, sixSeparating, { origin.x, origin.y, origin.z }, dps, gridRes );
        int2 xrange = context.xRangeInclusive();
        uint32_t nVoxels = 0;
        for( int x = xrange.x; x <= xrange.y; x++ )
        {
            int2 yrange = context.yRangeInclusive( x, dps );
            for( int y = yrange.x; y <= yrange.y; y++ )
            {
                int2 zrange = context.zRangeInclusive( x, y, dps, sixSeparating );
                for( int z = zrange.x; z <= zrange.y; z++ )
                {
                    float3 p = context.p( x, y, z, dps );
                    if( context.intersect( p ) )
                    {
                        nVoxels++;
                    }
                }
            }
        }
        uint32_t dstLocation = atomicAdd( counter, nVoxels );
        nVoxels = 0;

        for( int x = xrange.x; x <= xrange.y; x++ )
        {
            int2 yrange = context.yRangeInclusive( x, dps );
            for( int y = yrange.x; y <= yrange.y; y++ )
            {
                int2 zrange = context.zRangeInclusive( x, y, dps, sixSeparating );
                for( int z = zrange.x; z <= zrange.y; z++ )
                {
                    float3 p = context.p( x, y, z, dps );
                    if( context.intersect( p ) )
                    {
                        int3 c = context.i( x, y, z );
                        mortonVoxels[dstLocation + nVoxels] = encode2mortonCode_magicbits( c.x, c.y, c.z );

                        float3 bc = closestBarycentricCoordinateOnTriangle( v0, v1, v2, p );
                        float3 bColor = bc.x * c1 + bc.y * c2 + bc.z * c0;

						voxelColors[dstLocation + nVoxels] = { 
                            (uint8_t)( bColor.x * 255.0f + 0.5f ), 
                            (uint8_t)( bColor.y * 255.0f + 0.5f ), 
                            (uint8_t)( bColor.z * 255.0f + 0.5f ), 255 };
                        
                        nVoxels++;
                    }
                }
            }
        }
    }
}

// for countUnique -> unique
__device__ uint64_t g_iterator;

extern "C" __global__ void countUnique( const uint64_t* mortonVoxels, uint32_t n, uint32_t* counter )
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if( i < n )
    {
        if( i == 0 || mortonVoxels[i-1] != mortonVoxels[i]  )
        {
            atomicInc( counter, 0xFFFFFFFF );
        }
    }
	if( i == 0 )
	{
		g_iterator = 0;
    }
}

#define UNIQUE_BLOCK_SIZE 2048
#define UNIQUE_BLOCK_THREADS 64
#define UNIQUE_NUMBER_OF_ITERATION ( UNIQUE_BLOCK_SIZE / UNIQUE_BLOCK_THREADS )

template <int BLOCK_SIZE>
struct StreamCompaction64
{
    enum { 
        THREADS = 64,
        NUMBER_OF_STEPS = BLOCK_SIZE / THREADS
    };
	uint32_t gp;
	uint64_t leaderMasks[NUMBER_OF_STEPS];

    __device__ void init()
    {
		clearShared<NUMBER_OF_STEPS, THREADS, uint64_t>( leaderMasks, 0 );
		__syncthreads();
    }
	__device__ int steps() const { return NUMBER_OF_STEPS; }
	__device__ uint32_t itemIndex( int step ) const { return blockIdx.x * BLOCK_SIZE + step * THREADS + threadIdx.x; }
	__device__ void vote( int step )
    {
		atomicOr( &leaderMasks[step], 1llu << threadIdx.x );
    }

    // return global prefix
    __device__ uint32_t synchronize( uint64_t* iterator )
    {
		__syncthreads();

		if( threadIdx.x == 0 )
		{
			uint32_t prefix = 0;
			for( int i = 0; i < NUMBER_OF_STEPS; i++ )
			{
				prefix += __popcll( leaderMasks[i] );
			}

			uint64_t expected;
			uint64_t cur = *iterator;
			uint32_t globalPrefix = cur & 0xFFFFFFFF;
			do
			{
				expected = (uint64_t)globalPrefix + ( (uint64_t)( blockIdx.x ) << 32 );
				uint64_t newValue = (uint64_t)globalPrefix + prefix | ( (uint64_t)( blockIdx.x + 1 ) << 32 );
				cur = atomicCAS( iterator, expected, newValue );
				globalPrefix = cur & 0xFFFFFFFF;

			} while( cur != expected );

			gp = globalPrefix;
		}

		__syncthreads();

		return gp;
    }

	// return destination. If it is not voted, return -1
	__device__ uint32_t destination( int step, uint32_t* globalPrefix ) const
    {
		uint64_t mask = leaderMasks[step];
		bool voted = ( mask & ( 1llu << threadIdx.x ) ) != 0;
		uint64_t lowerMask = ( 1llu << threadIdx.x ) - 1;
		uint32_t offset = __popcll( mask & lowerMask );
        uint32_t d = *globalPrefix + offset;
		*globalPrefix += __popcll( mask );
		return voted ? d : -1;
    }
};

extern "C" __global__ void unique( const uint64_t* inputMortonVoxels, uint64_t* outputMortonVoxels, const uchar4* inputVoxelColors, uchar4* outputVoxelColors, uint32_t totalDumpedVoxels )
{
	__shared__ StreamCompaction64<UNIQUE_BLOCK_SIZE> streamCompaction;
	streamCompaction.init();

	for (int i = 0; i < streamCompaction.steps(); i++)
	{
		uint32_t itemIndex = streamCompaction.itemIndex( i );
		if( itemIndex < totalDumpedVoxels )
		{
			bool leader = itemIndex == 0 || inputMortonVoxels[itemIndex - 1] != inputMortonVoxels[itemIndex];
			if( leader )
			{
				streamCompaction.vote( i );
			}
		}
	}

	uint32_t globalPrefix = streamCompaction.synchronize( &g_iterator );

	for( int i = 0; i < streamCompaction.steps(); i++ )
	{
		uint32_t itemIndex = streamCompaction.itemIndex( i );
		if( itemIndex < totalDumpedVoxels )
		{
			uint32_t d = streamCompaction.destination( i, &globalPrefix );
			if( d != -1 ) // voted
			{
				uint64_t morton = inputMortonVoxels[itemIndex];
				outputMortonVoxels[d] = morton;

				int R = 0;
				int G = 0;
				int B = 0;
				int n = 0;
				for( int j = itemIndex; j < totalDumpedVoxels && inputMortonVoxels[j] == morton; j++ )
				{
					R += inputVoxelColors[j].x;
					G += inputVoxelColors[j].y;
					B += inputVoxelColors[j].z;
					n++;
				}
				uchar4 meanColor = {
					(uint8_t)( R / n ),
					(uint8_t)( G / n ),
					(uint8_t)( B / n ),
					255 };

				outputVoxelColors[d] = meanColor;
			}
		}
	}
}


__device__ uint64_t g_octreeIterator0;
__device__ uint64_t g_octreeIterator1;

extern "C" __global__ void octreeTaskInit( const uint64_t* inputMortonVoxels, uint32_t numberOfVoxels, OctreeTask* outputOctreeTasks, uint32_t* nPotentialMaxNodes )
{
	uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if( i < numberOfVoxels )
    {
		outputOctreeTasks[i].morton = inputMortonVoxels[i];
		outputOctreeTasks[i].child = -1;
		outputOctreeTasks[i].numberOfVoxels = 1;

		bool leader = i == 0 || ( inputMortonVoxels[i - 1] >> 3 ) != ( inputMortonVoxels[i] >> 3 );
		if( leader )
		{
			atomicInc( nPotentialMaxNodes, 0xFFFFFFFF );
		}
    }

    if( i == 0 )
    {
		g_octreeIterator0 = 0;
    }
}

#define BOTTOM_UP_BLOCK_SIZE 2048
extern "C" __global__ void bottomUpOctreeBuild( int iteration, const OctreeTask* inputOctreeTasks, uint32_t nInput, OctreeTask* outputOctreeTasks, uint32_t* nOutputTasks, OctreeNode* outputOctreeNodes, uint32_t* nOutputNodes )
{
	__shared__ StreamCompaction64<BOTTOM_UP_BLOCK_SIZE> streamCompaction;
	streamCompaction.init();

	for( int i = 0; i < streamCompaction.steps(); i++ )
	{
		uint32_t itemIndex = streamCompaction.itemIndex( i );
		if( itemIndex < nInput )
		{
			bool leader = itemIndex == 0 || inputOctreeTasks[itemIndex - 1].getMortonParent() != inputOctreeTasks[itemIndex].getMortonParent();
			if( leader )
			{
				streamCompaction.vote( i );
			}
		}
	}

	uint32_t globalPrefix = streamCompaction.synchronize( iteration % 2 == 0 ? &g_octreeIterator0 : &g_octreeIterator1 );

	for( int i = 0; i < streamCompaction.steps(); i++ )
	{
		uint32_t itemIndex = streamCompaction.itemIndex( i );
		if( itemIndex < nInput )
		{
			uint32_t d = streamCompaction.destination( i, &globalPrefix );
			if( d != -1 ) // voted
			{
				OctreeNode node;
				node.mask = 0;
				node.numberOfVoxels = 0;
				for( int j = 0; j < 8; j++ )
				{
					node.children[j] = -1;
				}

				// set child
				uint64_t mortonParent = inputOctreeTasks[itemIndex].getMortonParent();
				for( int j = itemIndex; j < nInput && inputOctreeTasks[j].getMortonParent() == mortonParent; j++ )
				{
					uint32_t space = inputOctreeTasks[j].morton & 0x7;
					node.mask |= ( 1 << space ) & 0xFF;
					node.children[space] = inputOctreeTasks[j].child;
					node.numberOfVoxels += inputOctreeTasks[j].numberOfVoxels;
				}
				
				uint32_t nodeIndex = atomicInc( nOutputNodes, 0xFFFFFFFF );
				outputOctreeNodes[nodeIndex] = node;

				outputOctreeTasks[d].morton = mortonParent;
				outputOctreeTasks[d].child = nodeIndex;
				outputOctreeTasks[d].numberOfVoxels = node.numberOfVoxels;

				atomicInc( nOutputTasks, 0xFFFFFFFF );
			}
		}
	}

    if( iteration % 2 == 0 )
	{
		if( threadIdx.x == 0 && blockIdx.x == 0 )
			g_octreeIterator1 = 0;
	}
	else
	{
		if( threadIdx.x == 0 && blockIdx.x == 0 )
			g_octreeIterator0 = 0;
	}
}