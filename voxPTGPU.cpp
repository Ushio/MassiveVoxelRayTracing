#include "pr.hpp"
#include <iostream>
#include <memory>
#include <set>

#include "Orochi/Orochi.h"
#include "Orochi/OrochiUtils.h"

#include "PathTracer.hpp"

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

	Config config;
	config.ScreenWidth = 1920;
	config.ScreenHeight = 1080;
	config.SwapInterval = 0;
	Initialize( config );

	Camera3D camera;
	camera.origin = { 4, 4, 4 };
	camera.lookat = { 0, 0, 0 };
	camera.zUp = false;

	static int frame = 0;

	//const char* input = "bunnyColor.abc";
	const char* input = "output.abc";
	AbcArchive ar;
	std::string errorMsg;
	ar.open( GetDataPath( input ), errorMsg );
	// std::shared_ptr<FScene> scene = ar.readFlat( 60, errorMsg );

	//scene->visitCamera( [&]( std::shared_ptr<const pr::FCameraEntity> cameraEntity )
	//{ 
	//	if( cameraEntity->visible() )
	//	{
	//		camera = cameraFromEntity( cameraEntity.get() );
	//	} 
	//} );

	//const char* input = "bunny.obj";
	//std::string errorMsg;
	//std::shared_ptr<FScene> scene = ReadWavefrontObj( GetDataPath( input ), errorMsg );

    std::vector<glm::vec3> vertices;
	std::vector<glm::vec3> vcolors;
	std::vector<glm::vec3> vemissions;
	// trianglesFlattened( scene, &vertices, &vcolors, &vemissions );

	PathTracer pt;
	pt.setup( stream, GetDataPath( "../voxKernel.cu" ).c_str(), GetDataPath( "../" ).c_str(), isNvidia );
	pt.resizeFrameBufferIfNeeded( stream, GetScreenWidth(), GetScreenHeight() );
	pt.loadHDRI( stream, "brown_photostudio_02_2k.hdr" );

	bool sixSeparating = true;
	int gridRes = 512;
	bool drawModel = false;
	bool buildAccelerationStructure = true;

	pr::ITexture* bgTexture = 0;

	SetDepthTest( true );

	bool sceneChanged = false;
	while( pr::NextFrame() == false )
	{
		if( IsImGuiUsingMouse() == false )
		{
			sceneChanged = sceneChanged || UpdateCameraBlenderLike( &camera );
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
		double cpuProcessMS = 0.0;
		if (buildAccelerationStructure)
		{
			Stopwatch sw;
			std::shared_ptr<FScene> scene = ar.readFlat( frame, errorMsg );

			//scene->visitCamera( [&]( std::shared_ptr<const pr::FCameraEntity> cameraEntity ){ 
			//if( cameraEntity->visible() )
			//{
			//	camera = cameraFromEntity( cameraEntity.get() );
			//} } );

			trianglesFlattened( scene, &vertices, &vcolors, &vemissions );

			glm::vec3 bbox_lower;
			glm::vec3 bbox_upper;
			getBoundingBox( vertices, &bbox_lower, &bbox_upper );
			glm::vec3 bbox_size = bbox_upper - bbox_lower;
			float dps = glm::max( glm::max( bbox_size.x, bbox_size.y ), bbox_size.z ) / (float)gridRes;

			cpuProcessMS = sw.elapsed() * 1000.0;

			OroStopwatch oroStream( stream );
			oroStream.start();
			pt.updateScene( vertices, vcolors, vemissions, stream, bbox_lower, dps, gridRes );
			oroStream.stop();
			buildMS = oroStream.getMs();
		}

		static Image2DRGBA8 image;
		image.allocate( GetScreenWidth(), GetScreenHeight() );

		pt.resizeFrameBufferIfNeeded( stream, GetScreenWidth(), GetScreenHeight() );

		double renderMS = 0;
		{
			if( sceneChanged )
			{
				pt.clearFrameBuffer( stream );
				sceneChanged = false;
			}

			OroStopwatch oroStream( stream );
			oroStream.start();
			pt.step( stream, camera );
			oroStream.stop();
			renderMS = oroStream.getMs();

			pt.toImageAsync( stream, &image );
			oroStreamSynchronize( stream );

			if( pt.getSteps() == 16 )
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
		ImGui::Text( "iteration = %d", pt.getSteps() );

		ImGui::SeparatorText( "Voxlizaiton" );
		ImGui::InputInt( "gridRes", &gridRes );
		if( ImGui::Button( "+", ImVec2( 100, 30 ) ) )
		{
			gridRes *= 2;
			sceneChanged = true;
		}
		if( ImGui::Button( "-", ImVec2( 100, 30 ) ) )
		{
			gridRes /= 2;
			sceneChanged = true;
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
		if( ImGui::SliderInt( "frame", &frame, 0, ar.frameCount() - 1 ) )
		{
			sceneChanged = true;
		}

		ImGui::Text( "build cpu(ms) = %f", cpuProcessMS );
		ImGui::Text( "build(ms) = %f", buildMS );
		
		ImGui::Text( "render(ms) = %f", renderMS );
		ImGui::Text( "voxels   = %lld", pt.getNumberOfVoxels() );
		ImGui::Text( "octree   = %lld byte", pt.getOctreeBytes() );
		
		if( ImGui::Button( "Save Image" ) )
		{
			image.saveAsPng( "render.png" );
		}

		ImGui::End();

		EndImGui();
	}
	pt.cleanUp();

	pr::CleanUp();
}