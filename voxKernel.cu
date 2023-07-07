#include "vectorMath.hpp"
#include "voxelization.hpp"

#include "voxCommon.hpp"
#include "IntersectorOctreeGPU.hpp"
#include "renderCommon.hpp"
#include "pmjSampler.hpp"

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

extern "C" __global__ void unique( const uint64_t* inputMortonVoxels, uint64_t* outputMortonVoxels, const uchar4* inputVoxelColors, uchar4* outputVoxelColors, uint32_t totalDumpedVoxels, uint64_t *iterator )
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

	uint32_t globalPrefix = streamCompaction.synchronize( iterator );

	for( int i = 0; i < streamCompaction.steps(); i++ )
	{
		uint32_t itemIndex = streamCompaction.itemIndex( i );
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


__device__ uint64_t g_octreeIterator0;
__device__ uint64_t g_octreeIterator1;

extern "C" __global__ void octreeTaskInit( const uint64_t* inputMortonVoxels, uint32_t numberOfVoxels, OctreeTask* outputOctreeTasks, uint32_t* taskCounters, uint32_t gridRes )
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
    if( i < numberOfVoxels )
    {
		uint64_t mortonL = inputMortonVoxels[max( i - 1, 0)];
		uint64_t mortonR = inputMortonVoxels[i];

		outputOctreeTasks[i].morton = mortonR;
		outputOctreeTasks[i].child = -1;
		outputOctreeTasks[i].numberOfVoxels = 1;

		int iteration = 0;
		while( 1 < ( gridRes >> iteration ) )
		{
			if( i == 0 || mortonL >> ( 3 * ( iteration + 1 ) ) != mortonR >> ( 3 * ( iteration + 1 ) ) )
			{
				atomicInc( &taskCounters[iteration], 0xFFFFFFFF );
			}
			iteration++;
		}
    }

    if( i == 0 )
    {
		g_octreeIterator0 = 0;
    }
}

#define BOTTOM_UP_BLOCK_SIZE 4096
extern "C" __global__ void bottomUpOctreeBuild( 
	int iteration,
	const OctreeTask* inputOctreeTasks, uint32_t nInput, 
	OctreeTask* outputOctreeTasks, 
	OctreeNode* outputOctreeNodes, uint32_t* nOutputNodes,
	uint32_t* lpBuffer, uint32_t lpSize )
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
		uint32_t d = streamCompaction.destination( i, &globalPrefix );

		uint8_t mask = 0;
		uint32_t numberOfVoxels = 0;
		uint32_t children[8];
		uint32_t nVoxelsPSum[8];
		for( int j = 0; j < 8; j++ )
		{
			children[j] = -1;
			nVoxelsPSum[j] = 0;
		}

		if( d != -1 ) // voted
		{
			// set child
			uint64_t mortonParent = inputOctreeTasks[itemIndex].getMortonParent();
			for( int j = itemIndex; j < nInput && inputOctreeTasks[j].getMortonParent() == mortonParent; j++ )
			{
				uint32_t space = inputOctreeTasks[j].morton & 0x7;
				mask |= ( 1 << space ) & 0xFF;
				children[space] = inputOctreeTasks[j].child;
				nVoxelsPSum[space] = inputOctreeTasks[j].numberOfVoxels;
			}

			// prefix scan exclusive
			for( int j = 0; j < 8; j++ )
			{
				uint32_t c = nVoxelsPSum[j];
				nVoxelsPSum[j] = numberOfVoxels;
				numberOfVoxels += c;
			}
			
			// Non DAG
#if !defined( ENABLE_GPU_DAG )
			uint32_t nodeIndex = atomicInc( nOutputNodes, 0xFFFFFFFF );
			outputOctreeNodes[nodeIndex].mask = mask;
			for( int j = 0; j < 8; j++ )
			{
				outputOctreeNodes[nodeIndex].children[j] = children[j];
				outputOctreeNodes[nodeIndex].nVoxelsPSum[j] = nVoxelsPSum[j];
			}

			outputOctreeTasks[d].morton = mortonParent;
			outputOctreeTasks[d].child = nodeIndex;
			outputOctreeTasks[d].numberOfVoxels = numberOfVoxels;
#endif
		}

#if defined( ENABLE_GPU_DAG )
		uint32_t nodeIndex = -1;

		MurmurHash32 h( 0 );
		h.combine( mask );
		for( int i = 0; i < 8; i++ )
			h.combine( children[i] );
		uint32_t home = h.getHash() % lpSize;

		bool done = d == -1;
#if defined( ITS )
		__syncwarp();
		for( int i = 0; __all_sync( 0xFFFFFFFF, done ) == false; i++ )
#else
		for( int i = 0; __all( done ) == false; i++ )
#endif
		{
			if( done )
			{
				continue;
			}

			int location = ( home + i ) % lpSize;
			uint32_t v = atomicCAS( &lpBuffer[location], 0, LP_LOCK );

			__threadfence();

			if( v == 0 ) // succeeded to lock
			{
				nodeIndex = atomicInc( nOutputNodes, 0xFFFFFFFF );
				outputOctreeNodes[nodeIndex].mask = mask;
				for( int j = 0; j < 8; j++ )
				{
					outputOctreeNodes[nodeIndex].children[j] = children[j];
					outputOctreeNodes[nodeIndex].nVoxelsPSum[j] = nVoxelsPSum[j];
				}

				__threadfence();

				atomicExch( &lpBuffer[location], nodeIndex | LP_OCCUPIED_BIT );

				done = true;
			}
			else if( v == LP_LOCK ) // someone is locking it
			{
				i--;
				continue; // try again
			}
			else
			{
				uint32_t otherNodeIndex = v & LP_VALUE_BIT;
				bool isEqual = outputOctreeNodes[otherNodeIndex].mask == mask;
				if( isEqual )
				for( int j = 0; j < 8; j++ )
				{
					if( outputOctreeNodes[otherNodeIndex].children[j] != children[j] )
					{
						isEqual = false;
						break;
					}
				}
				if( isEqual )
				{
					nodeIndex = otherNodeIndex;

					done = true;
				}
			}
		}

		if( d != -1 )
		{
			uint64_t mortonParent = inputOctreeTasks[itemIndex].getMortonParent();
			outputOctreeTasks[d].morton = mortonParent;
			outputOctreeTasks[d].child = nodeIndex;
			outputOctreeTasks[d].numberOfVoxels = numberOfVoxels;
		}
#endif
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

extern "C" __global__ void render( 
	uchar4* frameBuffer, int2 resolution, 
	uint32_t* taskCounter, StackElement* stackBuffer,
	CameraPinhole pinhole,
	IntersectorOctreeGPU intersector,
	int showVertexColor )
{
	__shared__ uint32_t taskIdx;

	StackElement* stack = stackBuffer + blockIdx.x * 32 * blockDim.x + threadIdx.x * 32;

	for (;; )
	{
		if( threadIdx.x == 0 )
		{
			taskIdx = atomicInc( taskCounter, 0xFFFFFFFF );
		}
		__syncthreads();

		uint32_t pixelIdx = taskIdx * blockDim.x + threadIdx.x;
		if( resolution.x * resolution.y <= pixelIdx )
		{
			break;
		}

		uint32_t x = pixelIdx % resolution.x;
		uint32_t y = pixelIdx / resolution.x;

		float3 ro, rd;
		pinhole.shoot( &ro, &rd, x, y, 0.5f, 0.5f, resolution.x, resolution.y );

		float t = MAX_FLOAT;
		int nMajor;
		uint32_t vIndex = 0;
		intersector.intersect( stack, ro, rd, &t, &nMajor, &vIndex );
		uchar4 colorOut = { 0, 0, 0, 255 };
		if( t != MAX_FLOAT )
		{
			if( showVertexColor )
			{
				colorOut = intersector.getVoxelColor( vIndex );
			}
			else
			{
				float3 hitN = getHitN( nMajor, rd );
				float3 color = ( hitN + float3{ 1.0f, 1.0f, 1.0f } ) * 0.5f;
				colorOut = { 255 * color.x + 0.5f, 255 * color.y + 0.5f, 255 * color.z + 0.5f, 255 };
			}
		}
		frameBuffer[y * resolution.x + x] = colorOut;
	}
}

extern "C" __global__ void HDRIstoreImportance( const float4* pixels, int2 resolution, double *sat, int cosWeighted, float3 axis )
{
	uint32_t pixelX = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t pixelY = blockIdx.y * blockDim.y + threadIdx.y;
	if( resolution.x <= pixelX || resolution.y <= pixelY )
	{
		return;
	}

	uint32_t pixelIdx = pixelY * resolution.x + pixelX;
	float dTheta = PI / (float)resolution.y;
	float dPhi = 2.0f * PI / (float)resolution.x;
	float theta = pixelY * dTheta;

	// dH = cos( theta ) - cos( theta + dTheta )
	//    = 2 sin( dTheta / 2 ) sin( dTheta / 2 + theta )
	float dH = 2.0f * INTRIN_SIN( dTheta * 0.5f ) * INTRIN_SIN( dTheta * 0.5f + theta );
	float dW = dPhi;
	float sr = dH * dW;
	float4 color = pixels[pixelIdx];

	float w = 1.0f;
	if( cosWeighted )
	{
		float sY = mix( INTRIN_COS( theta ), INTRIN_COS( theta + dTheta ), 0.5f );
		float phi = dPhi * ( (float)pixelX + 0.5f ) + PI;
		float sX = INTRIN_COS( phi );
		float sZ = INTRIN_SIN( phi );

		float sinTheta = INTRIN_SQRT( ss_max( 1.0f - sY * sY, 0.0f ) );
		float3 dirCenter = {
			sX * sinTheta,
			sY,
			sZ * sinTheta,
		};
		w = ss_max( dot( axis, dirCenter ), 0.0f );
	}

	sat[pixelIdx] = ( 0.2126f * color.x + 0.7152 * color.y + 0.0722 * color.z ) * sr * w;
}

template <class T, int NThreads>
__device__ inline T prefixSumInclusive( T prefix, T* sMemIO )
{
	for( uint32_t offset = 1; offset < NThreads; offset <<= 1 )
	{
		T x = sMemIO[threadIdx.x];

		if( offset <= threadIdx.x )
		{
			x += sMemIO[threadIdx.x - offset];
		}

		__syncthreads();

		sMemIO[threadIdx.x] = x;

		__syncthreads();
	}
	T sum = sMemIO[NThreads - 1];

	__syncthreads();

	sMemIO[threadIdx.x] += prefix;

	__syncthreads();

	return sum;
}

#define SAT_BLOCK_SIZE 512

extern "C" __global__ void buildSATh( int2 resolution, double* sat )
{
	__shared__ double s_mem[SAT_BLOCK_SIZE];
	int Y = blockIdx.x;

	double prefix = 0.0;
	for( int i = 0; i < resolution.x; i += SAT_BLOCK_SIZE )
	{
		int X = i + threadIdx.x;
		s_mem[threadIdx.x] = X < resolution.x ? sat[Y * resolution.x + X] : 0.0;

		__syncthreads();

		prefix += prefixSumInclusive<double, SAT_BLOCK_SIZE>( prefix, s_mem );

		if( X < resolution.x )
		{
			sat[Y * resolution.x + X] = s_mem[threadIdx.x];
		}
	}
}
extern "C" __global__ void buildSATv( int2 resolution, double* sat )
{
	__shared__ double s_mem[SAT_BLOCK_SIZE];
	int X = blockIdx.x;

	double prefix = 0.0;
	for( int i = 0; i < resolution.y; i += SAT_BLOCK_SIZE )
	{
		int Y = i + threadIdx.x;
		s_mem[threadIdx.x] = Y < resolution.y ? sat[Y * resolution.x + X] : 0.0;

		__syncthreads();

		prefix += prefixSumInclusive<double, SAT_BLOCK_SIZE>( prefix, s_mem );

		if( Y < resolution.y )
		{
			sat[Y * resolution.x + X] = s_mem[threadIdx.x];
		}
	}
}

extern "C" __global__ void buildSAT2u32( uint32_t* satU32, double* satF64, int n )
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	double sum = satF64[n - 1];
	if( i < n )
	{
		satU32[i] = (uint32_t)( satF64[i] / ( sum ) * (double)0xFFFFFFFFu );
	}
}

#define USE_PMJ 1
#define RENDER_NUMBER_OF_THREAD 64

extern "C" __global__ void renderPT(
	int iteration,
	float4* frameBuffer, int2 resolution,
	CameraPinhole pinhole,
	IntersectorOctreeGPU intersector,
	DynamicAllocatorGPU<StackElement> stackAllocator,
	HDRI hdri,
	PMJSampler pmj )
{
	uint32_t stackHandle;
	StackElement* stack = stackAllocator.acquire( &stackHandle );

	__shared__ uint32_t s_iterator;
	if( threadIdx.x == 0 )
	{
		s_iterator = 0;
	}
	__shared__ float localPixelValueXs[RENDER_NUMBER_OF_THREAD];
	__shared__ float localPixelValueYs[RENDER_NUMBER_OF_THREAD];
	__shared__ float localPixelValueZs[RENDER_NUMBER_OF_THREAD];
	localPixelValueXs[threadIdx.x] = 0.0f;
	localPixelValueYs[threadIdx.x] = 0.0f;
	localPixelValueZs[threadIdx.x] = 0.0f;

	__syncthreads();

	const int nBatchSpp = 16;

	for( int i = 0; i < nBatchSpp; i++ )
	{
		uint32_t taskIdx = atomicInc( &s_iterator, 0xFFFFFFFF );
		uint32_t localPixel = taskIdx / nBatchSpp;
		uint32_t localSpp = taskIdx % nBatchSpp;
		uint32_t pixelIdx = blockIdx.x * blockDim.x + localPixel;
		uint32_t spp = iteration * nBatchSpp + localSpp;

		if( pixelIdx < resolution.x * resolution.y )
		{
			uint32_t x = pixelIdx % resolution.x;
			uint32_t y = pixelIdx / resolution.x;

			MurmurHash32 hash( 0 );
			hash.combine( pixelIdx );
			uint32_t stream = hash.getHash();

#if defined( USE_PMJ )
			int dim = 0;
#define SAMPLE_2D() pmj.sample2d( spp, dim++, stream )
#else
			hash.combine( spp );
			PCG32 rng;
			rng.setup( 0, hash.getHash() );
#define SAMPLE_2D() float2{ uniformf( rng.nextU32() ), uniformf( rng.nextU32() ) }
#endif

			float2 cam_u01 = SAMPLE_2D();
			float3 ro, rd;
			pinhole.shoot( &ro, &rd, x, y, cam_u01.x, cam_u01.y, resolution.x, resolution.y );

			float3 T = { 1.0f, 1.0f, 1.0f };
			float3 L = {};
			for( int depth = 0; depth < 8; depth++ )
			{
				float t = MAX_FLOAT;
				int nMajor;
				uint32_t vIndex = 0;
				intersector.intersect( stack, ro, rd, &t, &nMajor, &vIndex );

				if( t == MAX_FLOAT )
				{
					//float I = ss_max( normalize( rd ).y, 0.0f ) * 3.0f;
					//float3 env = { I, I, I };
					if (depth == 0)
					{
						float3 env = hdri.sampleNearest( rd );
						L += T * env;
					}
					break;
				}

				float3 R = linearReflectance( intersector.getVoxelColor( vIndex ) );
				float3 hitN = getHitN( nMajor, rd );
				float3 hitP = ro + rd * t;

				{ // Explicit
					float2 u01 = SAMPLE_2D();
					float2 u23 = SAMPLE_2D();

					float3 dir;
					float3 emissive;
					float p;
					hdri.importanceSample( &dir, &emissive, &p, hitN, true, u01.x, u01.y, u23.x, u23.y );

					// no self intersection
					float t = MAX_FLOAT;
					int nMajor;
					uint32_t vIndex = 0;
					intersector.intersect( stack, hitP, dir, &t, &nMajor, &vIndex );
					if( t == MAX_FLOAT )
					{
						L += T * ( R / PI ) * ss_max( dot( hitN, dir ), 0.0f ) * emissive / p;
					}
				}

				T *= R;
			
				float2 u01 = SAMPLE_2D();
				float3 dir = sampleLambertian( u01.x, u01.y, hitN );

				ro = hitP; // no self intersection
				rd = dir;
			}

#undef SAMPLE_2D

			atomicAdd( &localPixelValueXs[localPixel], L.x );
			atomicAdd( &localPixelValueYs[localPixel], L.y );
			atomicAdd( &localPixelValueZs[localPixel], L.z );
		}
	}

	__syncthreads();

	uint32_t pixelIdx = blockIdx.x * blockDim.x + threadIdx.x;
	atomicAdd( &frameBuffer[pixelIdx].x, localPixelValueXs[threadIdx.x] );
	atomicAdd( &frameBuffer[pixelIdx].y, localPixelValueYs[threadIdx.x] );
	atomicAdd( &frameBuffer[pixelIdx].z, localPixelValueZs[threadIdx.x] );
	atomicAdd( &frameBuffer[pixelIdx].w, (float)nBatchSpp );
	stackAllocator.release( stackHandle );
}

extern "C" __global__ void renderResolve( uchar4* frameBufferU8, const float4* frameBufferF32, int n )
{
	uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if( i < n )
	{
		float4 value = frameBufferF32[i];
		uchar4 colorOut = {
			255 * INTRIN_POW( ss_min( value.x / value.w, 1.0f ), 1.0f / 2.2f ) + 0.5f,
			255 * INTRIN_POW( ss_min( value.y / value.w, 1.0f ), 1.0f / 2.2f ) + 0.5f,
			255 * INTRIN_POW( ss_min( value.z / value.w, 1.0f ), 1.0f / 2.2f ) + 0.5f,
			255 };
		frameBufferU8[i] = colorOut;
	}
}