#include "pr.hpp"
#include "prth.hpp"
#include <iostream>
#include <memory>
#include <set>

#include "Orochi/Orochi.h"
#include "Orochi/OrochiUtils.h"

#include "PathTracer.hpp"

uint32_t next_power_of_two(uint32_t v)
{
	v--;
	v |= v >> 1;
	v |= v >> 2;
	v |= v >> 4;
	v |= v >> 8;
	v |= v >> 16;
	v++;
	return v;
}

int main()
{
	using namespace pr;
	Stopwatch sw;
	SetDataDir( ExecutableDir() );

	if( oroInitialize( (oroApi)( ORO_API_HIP | ORO_API_CUDA ), 0 ) )
	{
		printf( "failed to init..\n" );
		return 0;
	}
	int deviceIdx = 0;
	int renderWidth = 1920 / 2;
	int renderHeigt = 1080 / 2;

	int totalFrames = 240;

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

	ThreadPool threadPool( 4 );
	TaskGroup taskGroup;

	// const char* input = "rtcamp.abc";
	const char* input = "output.abc";
	AbcArchive ar;
	std::string errorMsg;
	ar.open( GetDataPath( input ), errorMsg );

	std::vector<glm::vec3> vertices;
	std::vector<glm::vec3> vcolors;
	std::vector<glm::vec3> vemissions;

	PathTracer pt;
	pt.setup( stream, GetDataPath( "../voxKernel.cu" ).c_str(), GetDataPath( "../" ).c_str(), isNvidia );
	pt.resizeFrameBufferIfNeeded( stream, renderWidth, renderHeigt );
	pt.loadHDRI( stream, "brown_photostudio_02_2k.hdr" );

	glm::vec3 center = { -0.131793f, -1.40424f, -3.77277f };
	float boxWide = 15.0f;
	glm::vec3 origin = center - glm::vec3( boxWide * 0.5f );

	int fromRes = 256;
	int toRes = 2048;// 4096;
	for( int frame = 0; frame < 240; frame++ )
	{
		float dps = glm::mix( boxWide / fromRes, boxWide / toRes, (float)frame / totalFrames );
		int resolution = (int)ceil( boxWide / dps );
		int gridRes = next_power_of_two( resolution );

		Camera3D camera;
		std::shared_ptr<FScene> scene = ar.readFlat( frame, errorMsg );

		scene->visitCamera( [&]( std::shared_ptr<const pr::FCameraEntity> cameraEntity ) {
			if( cameraEntity->visible() )
			{
				camera = cameraFromEntity( cameraEntity.get() );
			} 
		} );

		trianglesFlattened( scene, &vertices, &vcolors, &vemissions );

		OroStopwatch swUpdate( stream );
		swUpdate.start();

		pt.updateScene( vertices, vcolors, vemissions, stream, origin, dps, gridRes );
		
		swUpdate.stop();

		OroStopwatch swRender( stream );
		swRender.start();

		pt.clearFrameBuffer( stream );

		for( int iteration = 0; iteration < 16; iteration++ )
		{
			pt.step( stream, camera );
		}

		swRender.stop();
		printf( "[frame %d] total( %.1f s ) / update %.3f / render %.3f\n", frame, sw.elapsed(), swUpdate.getMs(), swRender.getMs() );

		std::shared_ptr<Image2DRGBA8> image( new Image2DRGBA8() );
		image->allocate( renderWidth, renderHeigt );
		pt.toImageAsync( stream, image.get() );

		oroEvent imageEvent;
		oroEventCreateWithFlags( &imageEvent, oroEventDefault );
		oroEventRecord( imageEvent, stream );

		taskGroup.addElements( 1 );
		threadPool.enqueueTask( [frame, image, imageEvent, &taskGroup, device]() {
			oroCtx context = 0;
			oroCtxGetCurrent( &context );
			if( context == 0 )
			{
				oroCtxCreate( &context, 0, device );
				oroCtxSetCurrent( context );
			}

			oroEventSynchronize( imageEvent );
			oroEventDestroy( imageEvent );

			Stopwatch swSave;
			char output[256];
			sprintf( output, "render_%04d.png", frame );
			image->saveAsPngUncompressed( GetDataPath( output ).c_str() );
			taskGroup.doneElements( 1 );
		} );
	}
	taskGroup.waitForAllElementsToFinish();

	pt.cleanUp();

	return 0;
}