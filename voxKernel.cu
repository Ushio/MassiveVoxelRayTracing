#include "vectorMath.hpp"
#include "voxelization.hpp"

typedef unsigned long long uint64_t;
typedef unsigned int uint32_t;
typedef unsigned short uint16_t;
typedef unsigned char uint8_t;

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

extern "C" __global__ void voxCount( const float3 *vertices, const float3 *vcolors, uint32_t nTriangles, uint32_t* counter, float3 origin, float dps, uint32_t gridRes )
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
                        voxelColors[dstLocation + nVoxels] = { bColor.x * 255.0f + 0.5f, bColor.y * 255.0f + 0.5f, bColor.z * 255.0f + 0.5f, 255 };
                        
                        nVoxels++;
                    }
                }
            }
        }
    }
}
