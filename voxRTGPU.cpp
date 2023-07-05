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
int main()
{
	using namespace pr;
	SetDataDir( ExecutableDir() );

	if( oroInitialize( (oroApi)( ORO_API_HIP | ORO_API_CUDA ), 0 ) )
	{
		printf( "failed to init..\n" );
		return 0;
	}
	int deviceIdx = 2;

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
	static_assert( sizeof( glm::vec3 ) == sizeof( float3 ), "" );
	Buffer vertexBuffer( sizeof( float3 ) * vertices.size() );
	Buffer vcolorBuffer( sizeof( float3 ) * vcolors.size() );
	Buffer counterBuffer( sizeof( uint32_t ) );

	std::unique_ptr<Buffer> mortonVoxelsBuffer;
	std::unique_ptr<Buffer> voxelColorsBuffer;

	oroMemcpyHtoD( (oroDeviceptr)vertexBuffer.data(), vertices.data(), vertexBuffer.bytes() );
	oroMemcpyHtoD( (oroDeviceptr)vcolorBuffer.data(), vcolors.data(), vcolorBuffer.bytes() );

	glm::vec3 bbox_lower = glm::vec3( FLT_MAX );
	glm::vec3 bbox_upper = glm::vec3( -FLT_MAX );
	for( int i = 0; i < vertices.size(); i++ )
	{
		bbox_lower = glm::min( bbox_lower, vertices[i] );
		bbox_upper = glm::max( bbox_upper, vertices[i] );
	}

	bool sixSeparating = true;
	int gridRes = 512;
	bool drawModel = true;
	bool drawWire = false;
	bool showVertexColor = true;
	bool buildAccelerationStructure = true;
	bool renderParallel = false;

	enum DAGBUILD
	{
		DAGBUILD_NO,
		DAGBUILD_REF,
	};
	int dagBuild = DAGBUILD_REF;
	pr::ITexture* bgTexture = 0;
	std::shared_ptr<IntersectorOctree> octreeVoxel( new IntersectorOctree() );

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

		static std::vector<uint64_t> mortonVoxels;
		static std::vector<glm::u8vec4> voxelColors;
		Stopwatch sw;

		glm::vec3 origin = bbox_lower;
		glm::vec3 bbox_size = bbox_upper - bbox_lower;
		float dps = glm::max( glm::max( bbox_size.x, bbox_size.y ), bbox_size.z ) / (float)gridRes;

		if( buildAccelerationStructure )
		{
			oroMemsetD32Async( (oroDeviceptr)counterBuffer.data(), 0, 1, stream );
			//OroStopwatch oroStream( stream );
			//oroStream.start();
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
				voxKernel.launch( "voxCount", args, div_round_up64( nTriangles, 128 ), 1, 1, 128, 1, 1, stream );
			}


			uint32_t totalDumpedVoxels = 0;
			oroMemcpyDtoHAsync( &totalDumpedVoxels, (oroDeviceptr)counterBuffer.data(), sizeof( uint32_t ), stream );
			oroStreamSynchronize( stream );

			uint64_t mortonVoxelsBytes = sizeof( uint64_t ) * totalDumpedVoxels;
			if( !mortonVoxelsBuffer || mortonVoxelsBuffer->bytes() < mortonVoxelsBytes )
			{
				mortonVoxelsBuffer = std::unique_ptr<Buffer>( new Buffer( mortonVoxelsBytes ) );
				voxelColorsBuffer = std::unique_ptr<Buffer>( new Buffer( sizeof( uchar4 ) * totalDumpedVoxels ) );
			}

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
				args.add( voxelColorsBuffer->data() );
				voxKernel.launch( "voxelize", args, div_round_up64( nTriangles, 128 ), 1, 1, 128, 1, 1, stream );
			}

			{
				auto tmpBufferBytes = radixsort.getTemporaryBufferBytes( totalDumpedVoxels );
				Buffer tmpBuffer( tmpBufferBytes.getTemporaryBufferBytesForSortPairs() );
				radixsort.sortPairs( mortonVoxelsBuffer->data(), voxelColorsBuffer->data(), totalDumpedVoxels, tmpBuffer.data(), 0, 64, stream );
				oroStreamSynchronize( stream );
			}

			//oroStream.stop();
			//float voxelMs = oroStream.getMs();
			//printf( "voxelMs %f \n", voxelMs );

			// Compaction
			uint32_t numberOfVoxels = 0;
			{
				oroMemsetD32Async( (oroDeviceptr)counterBuffer.data(), 0, 1, stream );
				ShaderArgument args;
				args.add( mortonVoxelsBuffer->data() );
				args.add( totalDumpedVoxels );
				args.add( counterBuffer.data() );
				voxKernel.launch( "countUnique", args, div_round_up64( totalDumpedVoxels, 128 ), 1, 1, 128, 1, 1, stream );
				
				oroMemcpyDtoHAsync( &numberOfVoxels, (oroDeviceptr)counterBuffer.data(), sizeof( uint32_t ), stream );
				oroStreamSynchronize( stream );
			}

#define UNIQUE_BLOCK_SIZE 2048
#define UNIQUE_BLOCK_THREADS 64
			{
				auto outputMortonVoxelsBuffer = std::unique_ptr<Buffer>( new Buffer( sizeof( uint64_t ) * numberOfVoxels ) );
				auto outputVoxelColorsBuffer = std::unique_ptr<Buffer>( new Buffer( sizeof( uchar4 ) * numberOfVoxels ) );

				oroMemsetD32Async( (oroDeviceptr)counterBuffer.data(), 0, 1, stream );
				ShaderArgument args;
				args.add( mortonVoxelsBuffer->data() );
				args.add( outputMortonVoxelsBuffer->data() );
				args.add( voxelColorsBuffer->data() );
				args.add( outputVoxelColorsBuffer->data() );
				args.add( totalDumpedVoxels );
				voxKernel.launch( "unique", args, div_round_up64( totalDumpedVoxels, UNIQUE_BLOCK_SIZE ), 1, 1, UNIQUE_BLOCK_THREADS, 1, 1, stream );

				oroStreamSynchronize( stream );

				std::swap( mortonVoxelsBuffer, outputMortonVoxelsBuffer );
				std::swap( voxelColorsBuffer, outputVoxelColorsBuffer );
			}

			std::unique_ptr<Buffer> octreeTasksBuffer0( new Buffer( sizeof( OctreeTask ) * numberOfVoxels ) );
			std::unique_ptr<Buffer> octreeTasksBuffer1( new Buffer( sizeof( OctreeTask ) * numberOfVoxels ) );

			int nodeCapacity = 0;
			{
				oroMemsetD32Async( (oroDeviceptr)counterBuffer.data(), 0, 1, stream );

				ShaderArgument args;
				args.add( mortonVoxelsBuffer->data() );
				args.add( numberOfVoxels );
				args.add( octreeTasksBuffer0->data() );
				args.add( counterBuffer.data() );
				voxKernel.launch( "octreeTaskInit", args, div_round_up64( numberOfVoxels, 128 ), 1, 1, 128, 1, 1, stream );

				oroMemcpyDtoHAsync( &nodeCapacity, (oroDeviceptr)counterBuffer.data(), sizeof( uint32_t ), stream );
				oroStreamSynchronize( stream );
			}
			// nodeCapacity = numberOfVoxels;

			int numberOfNode = 0;
			std::unique_ptr<Buffer> nodeBuffer( new Buffer( sizeof( OctreeNode ) * nodeCapacity ) );

			int wide = gridRes;
			int iteration = 0;
			uint32_t nInput = numberOfVoxels;

			Buffer nOutputTasks( sizeof( uint32_t ) );
			oroMemsetD32Async( (oroDeviceptr)counterBuffer.data(), 0, 1, stream );

#define BOTTOM_UP_BLOCK_SIZE 2048
			while( 1 < ( gridRes >> iteration ) )
			{
				oroMemsetD32Async( (oroDeviceptr)nOutputTasks.data(), 0, 1, stream );

				ShaderArgument args;
				args.add( iteration );
				args.add( octreeTasksBuffer0->data() );
				args.add( nInput );
				args.add( octreeTasksBuffer1->data() );
				args.add( nOutputTasks.data() );
				args.add( nodeBuffer->data() );
				args.add( counterBuffer.data() ); // nOutputNodes
				voxKernel.launch( "bottomUpOctreeBuild", args, div_round_up64( nInput, BOTTOM_UP_BLOCK_SIZE ), 1, 1, 64, 1, 1, stream );
				 
				oroMemcpyDtoHAsync( &numberOfNode, (oroDeviceptr)counterBuffer.data(), sizeof( uint32_t ), stream );
				oroMemcpyDtoHAsync( &nInput, (oroDeviceptr)nOutputTasks.data(), sizeof( uint32_t ), stream );
				oroStreamSynchronize( stream );

				if( nodeCapacity < numberOfNode + nInput )
				{
					nodeCapacity = numberOfNode + nInput;
					std::unique_ptr<Buffer> newNodeBuffer( new Buffer( sizeof( OctreeNode ) * nodeCapacity ) );
					oroMemcpyDtoDAsync( (oroDeviceptr)newNodeBuffer->data(), (oroDeviceptr)nodeBuffer->data(), sizeof( OctreeNode ) * numberOfNode, stream );
					oroStreamSynchronize( stream );
					std::swap( nodeBuffer, newNodeBuffer );
				}

				std::swap( octreeTasksBuffer0, octreeTasksBuffer1 );

				iteration++;
			}
			printf( "r %f\n", (double)nodeCapacity / numberOfNode );

			// temporal construction
			octreeVoxel->m_lower = origin;
			octreeVoxel->m_upper = origin + glm::vec3( dps, dps, dps ) * (float)gridRes;
			octreeVoxel->m_nodes.resize( numberOfNode );
			oroMemcpyDtoHAsync( octreeVoxel->m_nodes.data(), (oroDeviceptr)nodeBuffer->data(), sizeof( OctreeNode ) * numberOfNode, stream );
			oroStreamSynchronize( stream );

			//mortonVoxels.resize( numberOfVoxels );
			//oroMemcpyDtoHAsync( mortonVoxels.data(), (oroDeviceptr)mortonVoxelsBuffer->data(), sizeof( uint64_t ) * numberOfVoxels, stream );
			voxelColors.resize( numberOfVoxels );
			oroMemcpyDtoHAsync( voxelColors.data(), (oroDeviceptr)voxelColorsBuffer->data(), sizeof( glm::u8vec4 ) * numberOfVoxels, stream );

			//oroStreamSynchronize( stream );

			// printf( "%d %d %f ms\n", counter, (int)mortonVoxels.size(), voxCountms );
		}
		double voxelizationTime = sw.elapsed();

		// mortonVoxels has some duplications but I don't care now.
		if( drawWire )
		{
			drawVoxelsWire( mortonVoxels, origin, dps, { 200, 200, 200 } );
		}

		double octreeBuildMS = 0.0;
		double embreeBuildMS = 0.0;
		#if 0
		if( buildAccelerationStructure )
		{
			//printf( "before %d\n", mortonVoxels.size() );
			// mergeVoxels( &mortonVoxels, &voxelColors );
			//printf( "after %d\n", mortonVoxels.size() );

			sw = Stopwatch();
			if( dagBuild == DAGBUILD_NO )
			{
				octreeVoxel->build( mortonVoxels, origin, dps, gridRes );
			}
			else if( dagBuild == DAGBUILD_REF )
			{
				octreeVoxel->buildDAGReference( mortonVoxels, origin, dps, gridRes );
			}
			octreeBuildMS = sw.elapsed() * 1000.0;
		}
		#endif

		#if 1
		Image2DRGBA8 image;
		image.allocate( GetScreenWidth(), GetScreenHeight() );

		CameraRayGenerator rayGenerator( GetCurrentViewMatrix(), GetCurrentProjMatrix(), image.width(), image.height() );

		sw = Stopwatch();

		auto renderLine = [&]( int j ) {
			for( int i = 0; i < image.width(); ++i )
			{
				glm::vec3 ro, rd;
				rayGenerator.shoot( &ro, &rd, i, j, 0.5f, 0.5f );
				glm::vec3 one_over_rd = glm::vec3( 1.0f ) / rd;

				float t = FLT_MAX;
				int nMajor;
				uint32_t vIndex = 0;
				octreeVoxel->intersect( ro, rd, &t, &nMajor, &vIndex );

				if( t != FLT_MAX )
				{
					glm::vec3 hitN = getHitN( nMajor, rd );

					if (showVertexColor)
					{
						image( i, j ) = voxelColors[vIndex];
					}
					else
					{
						glm::vec3 color = ( hitN + glm::vec3( 1.0f ) ) * 0.5f;
						image( i, j ) = { 255 * color.r, 255 * color.g, 255 * color.b, 255 };
					}
				}
				else
				{
					image( i, j ) = { 0, 0, 0, 255 };
				}
			}
		};
		if( renderParallel )
		{
			ParallelFor( image.height(), renderLine );
		}
		else
		{
			for( int j = 0; j < image.height(); ++j )
			{
				renderLine( j );
			}
		}
		double RT_MS = sw.elapsed();

		if( bgTexture == nullptr )
		{
			bgTexture = CreateTexture();
		}
		bgTexture->upload( image );
#endif

		PopGraphicState();
		EndCamera();

		BeginImGui();

		ImGui::SetNextWindowSize( { 500, 800 }, ImGuiCond_Once );
		ImGui::Begin( "Panel" );
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
		ImGui::Checkbox( "drawWire", &drawWire );

		ImGui::SeparatorText( "Acceleration" );

		if( 1000.0 < octreeBuildMS )
		{
			buildAccelerationStructure = false;
		}

		ImGui::Checkbox( "buildAccelerationStructure", &buildAccelerationStructure );
		ImGui::Checkbox( "showVertexColor( DAG Only )", &showVertexColor );
		ImGui::Text( "voxelization(ms) = %f", voxelizationTime * 1000.0 );
		ImGui::Text( "octree build(ms) = %f", octreeBuildMS );
		ImGui::Text( "embree build(ms) = %f", embreeBuildMS );
		ImGui::Text( "octree   = %lld byte", octreeVoxel->getMemoryConsumption() );
		ImGui::Text( "RT (ms) = %f", RT_MS );
		ImGui::Checkbox( "renderParallel", &renderParallel );
		ImGui::RadioButton( "DAG: none", &dagBuild, DAGBUILD_NO );
		ImGui::RadioButton( "DAG: reference", &dagBuild, DAGBUILD_REF );
		
		ImGui::End();

		EndImGui();
	}

	pr::CleanUp();
}