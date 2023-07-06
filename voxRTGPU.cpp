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
			args.add( intersectorOctreeGPU );
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

	intersectorOctreeGPU.cleanUp();

	pr::CleanUp();
}