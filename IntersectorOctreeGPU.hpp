#pragma once

#include "voxCommon.hpp"
#include "renderCommon.hpp"

#if !defined( __CUDACC__ ) && !defined( __HIPCC__ )
#include "tinyhipradixsort.hpp"
#include "Orochi/Orochi.h"
#include "Orochi/OrochiUtils.h"
#endif

#define UNIQUE_BLOCK_SIZE 2048
#define UNIQUE_BLOCK_THREADS 64
#define BOTTOM_UP_BLOCK_SIZE 4096

struct IntersectorOctreeGPU
{
	DEVICE IntersectorOctreeGPU() {}

#if !defined( __CUDACC__ ) && !defined( __HIPCC__ )
	void cleanUp()
	{
		if( m_vAttributeBuffer )
		{
			oroFree( (oroDeviceptr)m_vAttributeBuffer );
			m_vAttributeBuffer = 0;
		}
		//if (m_emissiveSurfaces)
		//{
		//	oroFree( (oroDeviceptr)m_emissiveSurfaces );
		//	m_emissiveSurfaces = 0;
		//}
		if( m_nodeBuffer )
		{
			oroFree( (oroDeviceptr)m_nodeBuffer );
			m_nodeBuffer = 0;
		}
	}

	void build(
		const std::vector<glm::vec3>& vertices, 
		const std::vector<glm::vec3>& vcolors, const std::vector<glm::vec3>& vemissions,
		Shader* voxKernel, oroStream stream,
		int gridRes )
	{
		if (__popcnt(gridRes) != 1)
		{
			abort();
		}

		thrs::RadixSort::Config rConfig;
		rConfig.keyType = thrs::KeyType::U64;
		rConfig.valueType = thrs::ValueType::U64;
		static thrs::RadixSort radixsort( {}, rConfig );

		// GPU buffer
		static_assert( sizeof( glm::vec3 ) == sizeof( float3 ), "" );
		Buffer vertexBuffer( sizeof( float3 ) * vertices.size() );
		Buffer vcolorBuffer( sizeof( float3 ) * vcolors.size() );
		Buffer vemissionBuffer( sizeof( float3 ) * vemissions.size() );
		Buffer counterBuffer( sizeof( uint32_t ) );

		oroMemcpyHtoD( (oroDeviceptr)vertexBuffer.data(), const_cast<glm::vec3*>( vertices.data() ), vertexBuffer.bytes() );
		oroMemcpyHtoD( (oroDeviceptr)vcolorBuffer.data(), const_cast<glm::vec3*>( vcolors.data() ), vcolorBuffer.bytes() );
		oroMemcpyHtoD( (oroDeviceptr)vemissionBuffer.data(), const_cast<glm::vec3*>( vemissions.data() ), vemissionBuffer.bytes() );

		glm::vec3 bbox_lower = glm::vec3( FLT_MAX );
		glm::vec3 bbox_upper = glm::vec3( -FLT_MAX );
		for( int i = 0; i < vertices.size(); i++ )
		{
			bbox_lower = glm::min( bbox_lower, vertices[i] );
			bbox_upper = glm::max( bbox_upper, vertices[i] );
		}

		glm::vec3 origin = bbox_lower;
		glm::vec3 bbox_size = bbox_upper - bbox_lower;
		float dps = glm::max( glm::max( bbox_size.x, bbox_size.y ), bbox_size.z ) / (float)gridRes;

		m_lower = { origin.x, origin.y, origin.z };
		m_upper = float3{ origin.x, origin.y, origin.z } + float3{ dps, dps, dps } * (float)gridRes;
		m_dps = dps;

		oroMemsetD32Async( (oroDeviceptr)counterBuffer.data(), 0, 1, stream );
		{
			uint32_t nTriangles = (uint32_t)( vertices.size() / 3 );
			ShaderArgument args;
			args.add( vertexBuffer.data() );
			args.add( nTriangles );
			args.add( counterBuffer.data() );
			args.add( origin );
			args.add( dps );
			args.add( (uint32_t)gridRes );
			voxKernel->launch( "voxCount", args, div_round_up64( nTriangles, 128 ), 1, 1, 128, 1, 1, stream );
		}

		uint32_t totalDumpedVoxels = 0;
		oroMemcpyDtoHAsync( &totalDumpedVoxels, (oroDeviceptr)counterBuffer.data(), sizeof( uint32_t ), stream );
		oroStreamSynchronize( stream );

		std::unique_ptr<Buffer> mortonVoxelsBuffer( new Buffer( sizeof( uint64_t ) * totalDumpedVoxels ) );
		if( m_vAttributeBuffer )
		{
			oroFree( (oroDeviceptr)m_vAttributeBuffer );
		}
		oroMalloc( (oroDeviceptr*)&m_vAttributeBuffer, sizeof( VoxelAttirb ) * totalDumpedVoxels );

		oroMemsetD32Async( (oroDeviceptr)counterBuffer.data(), 0, 1, stream );

		{
			uint32_t nTriangles = (uint32_t)( vertices.size() / 3 );
			ShaderArgument args;
			args.add( vertexBuffer.data() );
			args.add( vcolorBuffer.data() );
			args.add( vemissionBuffer.data() );
			args.add( nTriangles );
			args.add( counterBuffer.data() );
			args.add( origin );
			args.add( dps );
			args.add( (uint32_t)gridRes );
			args.add( mortonVoxelsBuffer->data() );
			args.add( m_vAttributeBuffer );
			voxKernel->launch( "voxelize", args, div_round_up64( nTriangles, 128 ), 1, 1, 128, 1, 1, stream );
		}

		{
			int sortBits = next_multiple( numberOfSortBitsMorton( gridRes ), 8 );

			auto tmpBufferBytes = radixsort.getTemporaryBufferBytes( totalDumpedVoxels );
			Buffer tmpBuffer( tmpBufferBytes.getTemporaryBufferBytesForSortPairs() );
			radixsort.sortPairs( mortonVoxelsBuffer->data(), (void*)m_vAttributeBuffer, totalDumpedVoxels, tmpBuffer.data(), 0, sortBits, stream );
			oroStreamSynchronize( stream );
		}

		{
			Buffer iteratorBuffer( sizeof( uint64_t ) );

			auto outputMortonVoxelsBuffer = std::unique_ptr<Buffer>( new Buffer( sizeof( uint64_t ) * totalDumpedVoxels ) );
			VoxelAttirb* outputVoxelAttribsBuffer = 0;
			oroMalloc( (oroDeviceptr*)&outputVoxelAttribsBuffer, sizeof( VoxelAttirb ) * totalDumpedVoxels );

			oroMemsetD32Async( (oroDeviceptr)counterBuffer.data(), 0, 1, stream );
			oroMemsetD32Async( (oroDeviceptr)iteratorBuffer.data(), 0, iteratorBuffer.bytes() / 4, stream );

			ShaderArgument args;
			args.add( mortonVoxelsBuffer->data() );
			args.add( outputMortonVoxelsBuffer->data() );
			args.add( m_vAttributeBuffer );
			args.add( outputVoxelAttribsBuffer );
			args.add( totalDumpedVoxels );
			args.add( iteratorBuffer.data() );
			voxKernel->launch( "unique", args, div_round_up64( totalDumpedVoxels, UNIQUE_BLOCK_SIZE ), 1, 1, UNIQUE_BLOCK_THREADS, 1, 1, stream );

			oroMemcpyDtoHAsync( &m_numberOfVoxels, (oroDeviceptr)iteratorBuffer.data(), sizeof( uint32_t ), stream );

			oroStreamSynchronize( stream );

			std::swap( mortonVoxelsBuffer, outputMortonVoxelsBuffer );

			oroFree( (oroDeviceptr)m_vAttributeBuffer );
			m_vAttributeBuffer = outputVoxelAttribsBuffer;
		}

		//// Gather Emissions
		//{
		//	oroMemsetD32Async( (oroDeviceptr)counterBuffer.data(), 0, 1, stream );
		//	ShaderArgument args;
		//	args.add( mortonVoxelsBuffer->data() );
		//	args.add( m_vAttributeBuffer );
		//	args.add( m_numberOfVoxels );
		//	args.add( counterBuffer.data() );
		//	voxKernel->launch( "countEmissiveSurfaces", args, div_round_up64( m_numberOfVoxels, 256 ), 1, 1, 256, 1, 1, stream );

		//	oroMemcpyDtoHAsync( &m_numberOfEmissiveSurfaces, (oroDeviceptr)counterBuffer.data(), sizeof( uint32_t ), stream );

		//	oroStreamSynchronize( stream );
		//}
		//{
		//	if( m_emissiveSurfaces )
		//	{
		//		oroFree( (oroDeviceptr)m_emissiveSurfaces );
		//	}
		//	oroMalloc( (oroDeviceptr*)&m_emissiveSurfaces, sizeof( EmissiveSurface ) * m_numberOfEmissiveSurfaces );
		//	oroMemsetD32Async( (oroDeviceptr)counterBuffer.data(), 0, 1, stream );

		//	LCGShuffler shuffler;
		//	if( m_numberOfEmissiveSurfaces )
		//	{
		//		PCG32 rng;
		//		rng.setup( 123, 0 );
		//		while( shuffler.tryInit( rng.nextU32(), rng.nextU32(), m_numberOfEmissiveSurfaces ) == false )
		//			;
		//	}

		//	ShaderArgument args;
		//	args.add( mortonVoxelsBuffer->data() );
		//	args.add( m_vAttributeBuffer );
		//	args.add( m_numberOfVoxels );
		//	args.add( counterBuffer.data() );
		//	args.add( m_emissiveSurfaces );
		//	args.add( float3{ origin.x, origin.y, origin.z } );
		//	args.add( dps );
		//	args.add( shuffler );
		//	voxKernel->launch( "gatherEmissiveSurfaces", args, div_round_up64( m_numberOfVoxels, 256 ), 1, 1, 256, 1, 1, stream );
		//}

		std::unique_ptr<Buffer> octreeTasksBuffer0( new Buffer( sizeof( OctreeTask ) * m_numberOfVoxels ) );

		int nIteration = 0;
		_BitScanForward( (unsigned long*)&nIteration, gridRes );
		Buffer taskCountersBuffer( sizeof( uint32_t ) * nIteration );
		oroMemsetD32Async( (oroDeviceptr)taskCountersBuffer.data(), 0, nIteration, stream );

		{
			ShaderArgument args;
			args.add( mortonVoxelsBuffer->data() );
			args.add( m_numberOfVoxels );
			args.add( octreeTasksBuffer0->data() );
			args.add( taskCountersBuffer.data() );
			args.add( gridRes );
			voxKernel->launch( "octreeTaskInit", args, div_round_up64( m_numberOfVoxels, 128 ), 1, 1, 128, 1, 1, stream );
		}

		std::vector<int> taskCounters( nIteration );
		oroMemcpyDtoHAsync( taskCounters.data(), (oroDeviceptr)taskCountersBuffer.data(), sizeof( uint32_t ) * nIteration, stream );
		oroStreamSynchronize( stream );

		std::unique_ptr<Buffer> octreeTasksBuffer1( new Buffer( sizeof( OctreeTask ) * taskCounters[0] /* the first outputs */ ) );

		int nTotalInternalNodes = 0;
		for( int i = 0; i < taskCounters.size(); i++ )
		{
#if defined( ENABLE_GPU_DAG )
			nTotalInternalNodes += i == 0 ? 256 /* DAG */ : taskCounters[i];
#else
			nTotalInternalNodes += taskCounters[i];
#endif
		}

		int lpSize = taskCounters[0];
		std::unique_ptr<Buffer> lpBuffer( new Buffer( sizeof( uint32_t ) * lpSize ) );

		if( m_nodeBuffer )
		{
			oroFree( (oroDeviceptr)m_nodeBuffer );
		}
		oroMalloc( (oroDeviceptr*)&m_nodeBuffer, sizeof( OctreeNode ) * nTotalInternalNodes );

		uint32_t nInput = m_numberOfVoxels;
		int wide = gridRes;
		int iteration = 0;

		oroMemsetD32Async( (oroDeviceptr)counterBuffer.data(), 0, 1, stream );


		while( 1 < ( gridRes >> iteration ) )
		{
			oroMemsetD32Async( (oroDeviceptr)lpBuffer->data(), 0, lpSize, stream );

			ShaderArgument args;
			args.add( iteration );
			args.add( octreeTasksBuffer0->data() );
			args.add( nInput );
			args.add( octreeTasksBuffer1->data() );
			args.add( m_nodeBuffer );
			args.add( counterBuffer.data() ); // nOutputNodes
			args.add( lpBuffer->data() );
			args.add( lpSize );
			voxKernel->launch( "bottomUpOctreeBuild", args, div_round_up64( nInput, BOTTOM_UP_BLOCK_SIZE ), 1, 1, 64, 1, 1, stream );

			nInput = taskCounters[iteration];

			std::swap( octreeTasksBuffer0, octreeTasksBuffer1 );

			iteration++;
		}

		oroMemcpyDtoHAsync( &m_numberOfNodes, (oroDeviceptr)counterBuffer.data(), sizeof( uint32_t ), stream );
		oroStreamSynchronize( stream );

		assert( m_numberOfNodes <= nTotalInternalNodes );
		// printf( "%d %d\n", m_numberOfNodes, nTotalInternalNodes );
	}
#else
	DEVICE void intersect(
		StackElement* stack,
		float3 ro,
		float3 rd,
		float* t, int* nMajor, uint32_t* vIndex) const
	{
		octreeTraverse_EfficientParametric( m_nodeBuffer, m_numberOfNodes - 1, stack, ro, rd, m_lower, m_upper, t, nMajor, vIndex );
	}
	DEVICE uchar4 getVoxelColor( uint32_t vIndex ) const 
	{
		return m_vAttributeBuffer[vIndex].color;
	}
	DEVICE float3 getVoxelEmission( uint32_t vIndex ) const
	{
		return linearReflectance( m_vAttributeBuffer[vIndex].emission ) * m_emissionScale;
	}
	//DEVICE uint32_t getNumberOfEmissiveSurfaces() const 
	//{ 
	//	return m_numberOfEmissiveSurfaces; 
	//}
	//DEVICE float getFaceWidth() const
	//{
	//	return m_dps;
	//}
	//DEVICE void getEmissiveFace( uint32_t indexOfSurface, float3* faceNormal, float3* faceCenter, float3* emission )
	//{
	//	EmissiveSurface f = m_emissiveSurfaces[indexOfSurface];
	//	float3 index = floorf( ( f.pivot - m_lower ) / m_dps );
	//	float3 voxelOrigin = m_lower + index * m_dps;
	//	float3 voxelCenter = voxelOrigin + float3{ m_dps * 0.5f, m_dps * 0.5f, m_dps * 0.5f };
	//	float3 dir = f.pivot - voxelCenter;
	//	float3 n = dir / ( m_dps / 4.0f );

	//	*faceNormal = n;
	//	*faceCenter = f.pivot + dir;
	//	*emission = linearReflectance( f.emission ) * m_emissionScale;
	//}
#endif
	VoxelAttirb* m_vAttributeBuffer = 0;
	//EmissiveSurface* m_emissiveSurfaces = 0;
	//uint32_t m_numberOfEmissiveSurfaces = 0;
	OctreeNode* m_nodeBuffer = 0;
	uint32_t m_numberOfNodes = 0;
	uint32_t m_numberOfVoxels = 0;
	float3 m_lower;
	float3 m_upper;
	float m_dps = 0.0f;

	float m_emissionScale = 2.0f;
};

template <class T>
struct DynamicAllocatorGPU
{
#if !defined( __CUDACC__ ) && !defined( __HIPCC__ )
	void setup( int numberOfBlock, int blockSize, int nElementPerThread, oroStream stream )
	{
		m_numberOfBlock = numberOfBlock;
		m_blockSize = blockSize;
		m_nElementPerThread = nElementPerThread;

		if( m_memory )
		{
			oroFree( (oroDeviceptr)m_memory );
		}
		if( m_locks )
		{
			oroFree( (oroDeviceptr)m_locks );
		}

		oroMalloc( (oroDeviceptr*)&m_memory, sizeof( T ) * m_nElementPerThread * m_blockSize * m_numberOfBlock );
		oroMalloc( (oroDeviceptr*)&m_locks, sizeof( uint32_t ) * m_numberOfBlock );
		oroMemsetD32Async( (oroDeviceptr)m_locks, 0, m_numberOfBlock, stream );
	}
	void cleanUp()
	{
		if( m_memory )
		{
			oroFree( (oroDeviceptr)m_memory );
			m_memory = 0;
		}
		if( m_locks )
		{
			oroFree( (oroDeviceptr)m_locks );
			m_locks = 0;
		}
	}
#else
	DEVICE T* acquire( uint32_t* handle )
	{
		__shared__ uint32_t s_handle;

		if( threadIdx.x == 0 )
		{
			MurmurHash32 h( 0x12345678 );
			h.combine( blockIdx.x );
			uint32_t i = h.getHash() % m_numberOfBlock;
			for(;;)
			{
				uint32_t v = atomicCAS( &m_locks[i], 0, 0xFFFFFFFF );
				if( v == 0 )
				{
					break;
				}
				i = ( i + 1 ) % m_numberOfBlock;
			}

			s_handle = i;
		}

		__syncthreads();
		__threadfence();

		uint32_t index = s_handle;
		*handle = index;

		return m_memory + index * m_nElementPerThread * m_blockSize + threadIdx.x * m_nElementPerThread;
	}
	DEVICE void release( uint32_t handle )
	{
		__threadfence();
		__syncthreads();

		if( threadIdx.x == 0 )
		{
			atomicExch( &m_locks[handle], 0 );
		}
	}
#endif

	T* m_memory = 0;
	uint32_t* m_locks = 0;
	int m_numberOfBlock = 0;
	int m_blockSize = 0;
	int m_nElementPerThread = 0;
};