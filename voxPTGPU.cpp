#include "pr.hpp"
#include <iostream>
#include <memory>
#include <set>

#include "voxUtil.hpp"
#include "Orochi/Orochi.h"
#include "Orochi/OrochiUtils.h"
#include "hipUtil.hpp"

#include "IntersectorOctreeGPU.hpp"
#include "voxCommon.hpp"
#include "renderCommon.hpp"
#include "pmjSampler.hpp"

#define RENDER_NUMBER_OF_THREAD 64

inline float3 toFloat3( glm::vec3 v )
{
	return { v.x, v.y, v.z };
}



int main()
{
	// PMJSampler pmjcpu;
	// pmjcpu.setup( false, 0 );

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
		// compilerArgs.push_back( "-G" );
		 compilerArgs.push_back( "--generate-line-info" );

		// ITS enabled
		compilerArgs.push_back( "--gpu-architecture=compute_70" );
	}
	else
	{
		compilerArgs.push_back( "-g" );
	}
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

	scene->visitCamera( [&]( std::shared_ptr<const pr::FCameraEntity> cameraEntity )
	{ 
		if( cameraEntity->visible() )
		{
			camera = cameraFromEntity( cameraEntity.get() );
		} 
	} );

	//const char* input = "bunny.obj";
	//std::string errorMsg;
	//std::shared_ptr<FScene> scene = ReadWavefrontObj( GetDataPath( input ), errorMsg );

    std::vector<glm::vec3> vertices;
	std::vector<glm::vec3> vcolors;
	trianglesFlattened( scene, &vertices, &vcolors );

	// GPU buffer
	Buffer counterBuffer( sizeof( uint32_t ) );
	IntersectorOctreeGPU intersectorOctreeGPU;
	DynamicAllocatorGPU<StackElement> stackAllocator;
	stackAllocator.setup( 16 * 256 /* numberOfBlock */, RENDER_NUMBER_OF_THREAD /* blockSize */, 32 /* nElementPerThread */, stream );

	Image2DRGBA32 hdriSrc;
	hdriSrc.loadFromHDR( "brown_photostudio_02_2k.hdr" );
	// hdriSrc.loadFromHDR( "modern_buildings_2_2k.hdr" );
	
	HDRI hdri;
	hdri.load( glm::value_ptr( *hdriSrc.data() ), hdriSrc.width(), hdriSrc.height(), &voxKernel, stream );

	PMJSampler pmj;
	pmj.setup( true, stream );

	int iteration = 0;
	std::unique_ptr<Buffer> frameBufferU8;
	std::unique_ptr<Buffer> frameBufferF32;

	bool sixSeparating = true;
	int gridRes = 512;
	bool drawModel = false;
	bool showVertexColor = true;
	bool buildAccelerationStructure = true;

	pr::ITexture* bgTexture = 0;

	SetDepthTest( true );

	while( pr::NextFrame() == false )
	{
		bool sceneChanged = false;
		if( IsImGuiUsingMouse() == false )
		{
			sceneChanged = UpdateCameraBlenderLike( &camera );
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

			auto frameBufferF32Bytes = image.width() * image.height() * sizeof( float4 );
			if( !frameBufferF32 || frameBufferF32->bytes() != frameBufferF32Bytes )
			{
				frameBufferF32 = std::unique_ptr<Buffer>( new Buffer( frameBufferF32Bytes ) );
				oroMemsetD32Async( (oroDeviceptr)frameBufferF32->data(), 0, frameBufferF32->bytes() / 4, stream );
			}
			auto frameBufferU8Bytes = image.width() * image.height() * sizeof( uchar4 );
			if( !frameBufferU8 || frameBufferU8->bytes() != frameBufferU8Bytes )
			{
				frameBufferU8 = std::unique_ptr<Buffer>( new Buffer( frameBufferU8Bytes ) );
			}

			if( sceneChanged )
			{
				iteration = 0;
				oroMemsetD32Async( (oroDeviceptr)frameBufferF32->data(), 0, frameBufferF32->bytes() / 4, stream );
			}

			{
				ShaderArgument args;
				args.add( iteration++ );
				args.add( frameBufferF32->data() );
				args.add<int2>( { image.width(), image.height() } );
				args.add( pinhole );
				args.add( intersectorOctreeGPU );
				args.add( stackAllocator );
				args.add( hdri );
				args.add( pmj );

				voxKernel.launch( "renderPT", args, div_round_up64( image.width() * image.height(), RENDER_NUMBER_OF_THREAD ), 1, 1, RENDER_NUMBER_OF_THREAD, 1, 1, stream );
			}
			{
				ShaderArgument args;
				args.add( frameBufferU8->data() );
				args.add( frameBufferF32->data() );
				args.add( image.width() * image.height() );
				voxKernel.launch( "renderResolve", args, div_round_up64( image.width() * image.height(), 128 ), 1, 1, 128, 1, 1, stream );
			}

			oroStream.stop();
			renderMS = oroStream.getMs();

			oroMemcpyDtoHAsync( image.data(), (oroDeviceptr)frameBufferU8->data(), image.width() * image.height() * sizeof( uchar4 ), stream );
			oroStreamSynchronize( stream );

			if( iteration == 16 )
			{
				image.saveAsPng( "render_first.png" );
			}
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
		ImGui::Text( "iteration = %d", iteration );

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
		
		if( ImGui::Button( "Save Image" ) )
		{
			image.saveAsPng( "render.png" );
		}

		ImGui::End();

		EndImGui();
	}

	intersectorOctreeGPU.cleanUp();
	stackAllocator.cleanUp();

	pr::CleanUp();
}