#include "morton.hpp"
#include "pr.hpp"
#include "voxelization.hpp"
#include "IntersectorOctree.hpp"
#include <iostream>
#include <memory>
#include <set>

#include "voxUtil.hpp"
#include "Orochi/Orochi.h"
#include "Orochi/OrochiUtils.h"
#include "hipUtil.hpp"

#include "tinyhipradixsort.hpp"

#include "voxCommon.hpp"

void mergeVoxels( std::vector<uint64_t>* keys, std::vector<glm::u8vec4> *values )
{
	std::map<uint64_t, glm::uvec4> voxels;
	for (int i = 0; i < keys->size(); i++)
	{
		auto key = ( *keys )[i];
		auto value = ( *values )[i];
		auto it = voxels.find( key );
		if( it == voxels.end() )
		{
			voxels[key] = { value.x, value.y, value.z, 1 };
		}
		else
		{
			it->second += glm::vec4{ value.x, value.y, value.z, 1 };
		}
	}
	keys->clear();
	values->clear();

	for( auto kv : voxels )
	{
		keys->push_back( kv.first );
		auto v = glm::u8vec4( kv.second / kv.second.w );

		values->push_back( glm::u8vec4( kv.second / kv.second.w ) );
	}
}
inline float3 toFloat3( glm::vec3 v )
{
	return { v.x, v.y, v.z };
}

struct IntersectorOctreeGPU
{
	IntersectorOctreeGPU() {}
	~IntersectorOctreeGPU()
	{
		if( m_vcolorBuffer )
		{
			oroFree( (oroDeviceptr)m_vcolorBuffer );
		}
		if( m_nodeBuffer )
		{
			oroFree( (oroDeviceptr)m_nodeBuffer );
		}
	}
	void build( 
		const std::vector<glm::vec3>& vertices, const std::vector<glm::vec3>& vcolors, 
		Shader* voxKernel, oroStream stream,
		int gridRes
	)
	{
		thrs::RadixSort::Config rConfig;
		rConfig.keyType = thrs::KeyType::U64;
		rConfig.valueType = thrs::ValueType::U32;
		static thrs::RadixSort radixsort( {}, rConfig );

		// GPU buffer
		static_assert( sizeof( glm::vec3 ) == sizeof( float3 ), "" );
		Buffer vertexBuffer( sizeof( float3 ) * vertices.size() );
		Buffer vcolorBuffer( sizeof( float3 ) * vcolors.size() );
		Buffer counterBuffer( sizeof( uint32_t ) );

		oroMemcpyHtoD( (oroDeviceptr)vertexBuffer.data(), const_cast<glm::vec3*>( vertices.data() ), vertexBuffer.bytes() );
		oroMemcpyHtoD( (oroDeviceptr)vcolorBuffer.data(), const_cast<glm::vec3*>( vcolors.data() ), vcolorBuffer.bytes() );

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

		oroMemsetD32Async( (oroDeviceptr)counterBuffer.data(), 0, 1, stream );
		{
			uint32_t nTriangles = (uint32_t)( vertices.size() / 3 );
			ShaderArgument args;
			args.add( vertexBuffer.data() );
			args.add( vcolorBuffer.data() );
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
		if( m_vcolorBuffer )
		{
			oroFree( (oroDeviceptr)m_vcolorBuffer );
		}
		oroMalloc( (oroDeviceptr*)&m_vcolorBuffer, sizeof( uchar4 ) * totalDumpedVoxels );

		oroMemsetD32Async( (oroDeviceptr)counterBuffer.data(), 0, 1, stream );

		{
			uint32_t nTriangles = (uint32_t)( vertices.size() / 3 );
			ShaderArgument args;
			args.add( vertexBuffer.data() );
			args.add( vcolorBuffer.data() );
			args.add( nTriangles );
			args.add( counterBuffer.data() );
			args.add( origin );
			args.add( dps );
			args.add( (uint32_t)gridRes );
			args.add( mortonVoxelsBuffer->data() );
			args.add( m_vcolorBuffer );
			voxKernel->launch( "voxelize", args, div_round_up64( nTriangles, 128 ), 1, 1, 128, 1, 1, stream );
		}

		{
			auto tmpBufferBytes = radixsort.getTemporaryBufferBytes( totalDumpedVoxels );
			Buffer tmpBuffer( tmpBufferBytes.getTemporaryBufferBytesForSortPairs() );
			radixsort.sortPairs( mortonVoxelsBuffer->data(), (void*)m_vcolorBuffer, totalDumpedVoxels, tmpBuffer.data(), 0, 64, stream );
			oroStreamSynchronize( stream );
		}

#define UNIQUE_BLOCK_SIZE 2048
#define UNIQUE_BLOCK_THREADS 64

		uint32_t numberOfVoxels = 0;
		{
			Buffer iteratorBuffer( sizeof( uint64_t ) );

			auto outputMortonVoxelsBuffer = std::unique_ptr<Buffer>( new Buffer( sizeof( uint64_t ) * totalDumpedVoxels ) );
			uchar4* outputVoxelColorsBuffer = 0;
			oroMalloc( (oroDeviceptr*)&outputVoxelColorsBuffer, sizeof( uchar4 ) * totalDumpedVoxels );

			oroMemsetD32Async( (oroDeviceptr)counterBuffer.data(), 0, 1, stream );
			oroMemsetD32Async( (oroDeviceptr)iteratorBuffer.data(), 0, iteratorBuffer.bytes() / 4, stream );

			ShaderArgument args;
			args.add( mortonVoxelsBuffer->data() );
			args.add( outputMortonVoxelsBuffer->data() );
			args.add( m_vcolorBuffer );
			args.add( outputVoxelColorsBuffer );
			args.add( totalDumpedVoxels );
			args.add( iteratorBuffer.data() );
			voxKernel->launch( "unique", args, div_round_up64( totalDumpedVoxels, UNIQUE_BLOCK_SIZE ), 1, 1, UNIQUE_BLOCK_THREADS, 1, 1, stream );

			oroMemcpyDtoHAsync( &numberOfVoxels, (oroDeviceptr)iteratorBuffer.data(), sizeof( uint32_t ), stream );

			oroStreamSynchronize( stream );

			std::swap( mortonVoxelsBuffer, outputMortonVoxelsBuffer );

			oroFree( (oroDeviceptr)m_vcolorBuffer );
			m_vcolorBuffer = outputVoxelColorsBuffer;
		}

		std::unique_ptr<Buffer> octreeTasksBuffer0( new Buffer( sizeof( OctreeTask ) * numberOfVoxels ) );

		int nIteration = 0;
		_BitScanForward( (unsigned long*)&nIteration, gridRes );
		Buffer taskCountersBuffer( sizeof( uint32_t ) * nIteration );
		oroMemsetD32Async( (oroDeviceptr)taskCountersBuffer.data(), 0, nIteration, stream );

		{
			ShaderArgument args;
			args.add( mortonVoxelsBuffer->data() );
			args.add( numberOfVoxels );
			args.add( octreeTasksBuffer0->data() );
			args.add( taskCountersBuffer.data() );
			args.add( gridRes );
			voxKernel->launch( "octreeTaskInit", args, div_round_up64( numberOfVoxels, 128 ), 1, 1, 128, 1, 1, stream );
		}

		std::vector<int> taskCounters( nIteration );
		oroMemcpyDtoHAsync( taskCounters.data(), (oroDeviceptr)taskCountersBuffer.data(), sizeof( uint32_t ) * nIteration, stream );
		oroStreamSynchronize( stream );

		std::unique_ptr<Buffer> octreeTasksBuffer1( new Buffer( sizeof( OctreeTask ) * taskCounters[0] /* the first outputs */ ) );

		int nTotalInternalNodes = 0;
		for( int i = 0; i < taskCounters.size() ; i++ )
		{
			nTotalInternalNodes += i == 0 ? 256 /* DAG */ : taskCounters[i];
		}

		int lpSize = taskCounters[0];
		std::unique_ptr<Buffer> lpBuffer( new Buffer( sizeof( uint32_t ) * lpSize ) );

		if( m_nodeBuffer )
		{
			oroFree( (oroDeviceptr)m_nodeBuffer );
		}
		oroMalloc( (oroDeviceptr*)&m_nodeBuffer, sizeof( OctreeNode ) * nTotalInternalNodes );

		uint32_t nInput = numberOfVoxels;
		int wide = gridRes;
		int iteration = 0;

		oroMemsetD32Async( (oroDeviceptr)counterBuffer.data(), 0, 1, stream );

#define BOTTOM_UP_BLOCK_SIZE 4096
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

	IntersectorOctreeGPU( const IntersectorOctreeGPU& ) = delete;
	void operator=( const IntersectorOctreeGPU& ) = delete;

	uchar4* m_vcolorBuffer = 0;
	OctreeNode* m_nodeBuffer = 0;
	uint32_t m_numberOfNodes = 0;
	float3 m_lower;
	float3 m_upper;
};

int main()
{
	using namespace pr;
	SetDataDir( ExecutableDir() );

	if( oroInitialize( (oroApi)( ORO_API_HIP | ORO_API_CUDA ), 0 ) )
	{
		printf( "failed to init..\n" );
		return 0;
	}
	int deviceIdx = 0;

	oroError err;
	err = oroInit( 0 );
	oroDevice device;
	err = oroDeviceGet( &device, deviceIdx );
	oroCtx ctx;
	err = oroCtxCreate( &ctx, 0, device );
	oroCtxSetCurrent( ctx );

	oroStream stream = 0;
	oroStreamCreate( &stream );
	oroDeviceProp props;
	oroGetDeviceProperties( &props, device );

	bool isNvidia = oroGetCurAPI( 0 ) & ORO_API_CUDADRIVER;

	printf( "Device: %s\n", props.name );
	printf( "Cuda: %s\n", isNvidia ? "Yes" : "No" );

	std::vector<char> voxSrc;
	loadFileAsVector( &voxSrc, GetDataPath( "../voxKernel.cu" ).c_str() );
	voxSrc.push_back( '\0' );

	std::vector<std::string> compilerArgs;
	compilerArgs.push_back( "-I" + GetDataPath( "../" ) ); 

	if( isNvidia )
	{
		compilerArgs.push_back( "--generate-line-info" );

		// ITS enabled
		compilerArgs.push_back( "--gpu-architecture=compute_70" );
	}
	else
	{
		compilerArgs.push_back( "-g" );
	}
	thrs::RadixSort::Config rConfig;
	rConfig.keyType = thrs::KeyType::U64;
	rConfig.valueType = thrs::ValueType::U32;
	thrs::RadixSort radixsort( compilerArgs, rConfig );

	Shader voxKernel( voxSrc.data(), "voxKernel.cu", compilerArgs );

	Config config;
	config.ScreenWidth = 1920;
	config.ScreenHeight = 1080;
	config.SwapInterval = 0;
	Initialize( config );

	Camera3D camera;
	camera.origin = { 4, 4, 4 };
	camera.lookat = { 0, 0, 0 };
	camera.zUp = false;

	const char* input = "bunnyColor.abc";
	AbcArchive ar;
	std::string errorMsg;
	ar.open( GetDataPath( input ), errorMsg );
	std::shared_ptr<FScene> scene = ar.readFlat( 0, errorMsg );

	//const char* input = "bunny.obj";
	//std::string errorMsg;
	//std::shared_ptr<FScene> scene = ReadWavefrontObj( GetDataPath( input ), errorMsg );

    std::vector<glm::vec3> vertices;
	std::vector<glm::vec3> vcolors;
	trianglesFlattened( scene, &vertices, &vcolors );

	// GPU buffer
	Buffer counterBuffer( sizeof( uint32_t ) );
	IntersectorOctreeGPU intersectorOctreeGPU;

	std::unique_ptr<Buffer> stackBuffer;
	std::unique_ptr<Buffer> frameBuffer;

	bool sixSeparating = true;
	int gridRes = 512;
	bool drawModel = true;
	bool showVertexColor = true;
	bool buildAccelerationStructure = true;

	pr::ITexture* bgTexture = 0;

	SetDepthTest( true );

	while( pr::NextFrame() == false )
	{
		if( IsImGuiUsingMouse() == false )
		{
			UpdateCameraBlenderLike( &camera );
		}
		if( bgTexture )
		{
			ClearBackground( bgTexture );
		}
		else
		{
			ClearBackground( 0.1f, 0.1f, 0.1f, 1 );
		}

		BeginCamera( camera );

		PushGraphicState();

		DrawGrid( GridAxis::XZ, 1.0f, 10, { 128, 128, 128 } );
		DrawXYZAxis( 1.0f );

		if( drawModel )
		{
			PrimBegin( pr::PrimitiveMode::Lines );
			for( int i = 0; i < vertices.size(); i += 3 )
			{
				uint32_t indices[3];
				for( int j = 0; j < 3; j++ )
				{
					indices[j] = pr::PrimVertex( vertices[i + j], { 255, 255, 255 } );
				}
				for( int j = 0; j < 3; j++ )
				{
					pr::PrimIndex( indices[j] );
					pr::PrimIndex( indices[( j + 1 ) % 3] );
				}
			}
			PrimEnd();
		}
		double buildMS = 0.0;
		if (buildAccelerationStructure)
		{
			OroStopwatch oroStream( stream );
			oroStream.start();
			intersectorOctreeGPU.build( vertices, vcolors, &voxKernel, stream, gridRes );
			oroStream.stop();
			buildMS = oroStream.getMs();
		}

		Image2DRGBA8 image;
		image.allocate( GetScreenWidth(), GetScreenHeight() );

		CameraPinhole pinhole;
		pinhole.initFromPerspective( GetCurrentViewMatrix(), GetCurrentProjMatrix() );

		double renderMS = 0;
		{
			OroStopwatch oroStream( stream );
			oroStream.start();

			int nBlock = 16 * 256;
			int nThreads = 256;

			auto frameBufferBytes = image.width() * image.height() * sizeof( uchar4 );
			if( !frameBuffer || frameBuffer->bytes() != frameBufferBytes )
			{
				frameBuffer = std::unique_ptr<Buffer>( new Buffer( frameBufferBytes ) );
			}
			if( !stackBuffer )
			{
				stackBuffer = std::unique_ptr<Buffer>( new Buffer( sizeof( StackElement ) * 32 * nThreads * nBlock ) );
			}
			oroMemsetD32Async( (oroDeviceptr)counterBuffer.data(), 0, 1, stream );

			ShaderArgument args;
			args.add( frameBuffer->data() );
			args.add<int2>( { image.width(), image.height() } );
			args.add( counterBuffer.data() );
			args.add( stackBuffer->data() );
			args.add( pinhole );
			args.add( intersectorOctreeGPU.m_nodeBuffer );
			args.add( intersectorOctreeGPU.m_numberOfNodes - 1 );
			args.add( intersectorOctreeGPU.m_vcolorBuffer );
			args.add( intersectorOctreeGPU.m_lower );
			args.add( intersectorOctreeGPU.m_upper );
			args.add( showVertexColor ? 1 : 0 );
			
			voxKernel.launch( "render", args, nBlock, 1, 1, nThreads, 1, 1, stream );
			
			oroStream.stop();
			renderMS = oroStream.getMs();

			oroMemcpyDtoHAsync( image.data(), (oroDeviceptr)frameBuffer->data(), frameBuffer->bytes(), stream );
			oroStreamSynchronize( stream );
		}

		if( bgTexture == nullptr )
		{
			bgTexture = CreateTexture();
		}
		bgTexture->upload( image );


		PopGraphicState();
		EndCamera();

		BeginImGui();

		ImGui::SetNextWindowSize( { 500, 800 }, ImGuiCond_Once );
		ImGui::Begin( "Panel" );
		ImGui::Text( "device = %s", props.name );
		ImGui::Text( "fps = %f", GetFrameRate() );

		ImGui::SeparatorText( "Voxlizaiton" );
		ImGui::InputInt( "gridRes", &gridRes );
		if( ImGui::Button( "+", ImVec2( 100, 30 ) ) )
		{
			gridRes *= 2;
		}
		if( ImGui::Button( "-", ImVec2( 100, 30 ) ) )
		{
			gridRes /= 2;
		}
		ImGui::Checkbox( "sixSeparating", &sixSeparating );
		
		ImGui::SeparatorText( "Drawing" );
		ImGui::Checkbox( "drawModel", &drawModel );

		ImGui::SeparatorText( "Acceleration" );

		if( 1000.0 < buildMS )
		{
			buildAccelerationStructure = false;
		}

		ImGui::Checkbox( "buildAccelerationStructure", &buildAccelerationStructure );
		ImGui::Checkbox( "showVertexColor", &showVertexColor );
		ImGui::Text( "build(ms) = %f", buildMS );
		ImGui::Text( "render(ms) = %f", renderMS );
		ImGui::Text( "octree   = %lld byte", intersectorOctreeGPU.m_numberOfNodes * sizeof( OctreeNode ) );
		
		ImGui::End();

		EndImGui();
	}

	pr::CleanUp();
}