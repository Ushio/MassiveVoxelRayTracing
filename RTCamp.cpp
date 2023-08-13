#include "pr.hpp"
#include "prth.hpp"
#include <iostream>
#include <memory>
#include <set>
#include <concurrent_queue.h>

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

int main( int argc, char* argv[] )
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
	int renderWidth = 1440;
	int renderHeigt = 900;
	int beginFrame = 0;
	int endFrame = 240;

	if( 4 <= argc )
	{
		beginFrame = atoi( argv[2] );
		endFrame = atoi( argv[3] );
		printf( "frame specified: %d %d\n", beginFrame, endFrame );
	}

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

	ThreadPool threadPool( 1 );
	TaskGroup taskGroup;

	// const char* input = "rtcamp.abc";
	const char* input = "output.abc";
	AbcArchive ar;
	std::string errorMsg;
	ar.open( GetDataPath( input ), errorMsg );

	// Scene data
	Camera3D camera;
	float lensR = 0.1f;
	float focus = 1.0f;
	std::vector<glm::vec3> vertices;
	std::vector<glm::vec3> vcolors;
	std::vector<glm::vec3> vemissions;

	auto loadSceneFrame = [&ar, &camera, &focus, &vertices, &vcolors, &vemissions]( int frame )
	{
		std::string errorMsg;
		std::shared_ptr<FScene> scene = ar.readFlat( frame, errorMsg );

		scene->visitCamera( [&]( std::shared_ptr<const pr::FCameraEntity> cameraEntity ) {
			if( cameraEntity->visible() )
			{
				camera = cameraFromEntity( cameraEntity.get() );
				focus = cameraEntity->focusDistance();
			} } );

		trianglesFlattened( scene, &vertices, &vcolors, &vemissions );
	};
	loadSceneFrame( beginFrame ); // load first frame

	PathTracer pt;
	if( FILE* fp = fopen( GetDataPath( "../voxKernel.cu" ).c_str(), "rb" ) )
	{
		fclose( fp );
		pt.setup( stream, GetDataPath( "../voxKernel.cu" ).c_str(), GetDataPath( "../" ).c_str(), isNvidia );
	}
	else
	{
		pt.setup( stream, GetDataPath( "./voxKernel.cu" ).c_str(), GetDataPath( "./" ).c_str(), isNvidia );
	}

	pt.resizeFrameBufferIfNeeded( stream, renderWidth, renderHeigt );
	pt.loadHDRI( stream, "monks_forest_2k.hdr", "monks_forest_2k_primary.hdr" );

	// reading buffer
	Concurrency::concurrent_queue<Buffer*> frameBufferPool;
	for( int i = 0; i < 4; i++ )
	{
		frameBufferPool.push( new Buffer( (uint64_t)renderWidth * renderHeigt * sizeof( uchar4 ) ) );
	}

	glm::vec3 center = { 3.16434f, -1.40424f, -3.77277f };
	float boxWide = 20.0f;
	glm::vec3 origin = center - glm::vec3( boxWide * 0.5f );

	int fromRes = 400;
	int toRes = 8192; // 4096;
	for( int frame = beginFrame; frame < endFrame; frame++ )
	{
		float dps = glm::mix( boxWide / fromRes, boxWide / toRes, (float)frame / totalFrames );
		int resolution = (int)ceil( boxWide / dps );
		int gridRes = next_power_of_two( resolution );

		OroStopwatch swUpdate( stream );
		swUpdate.start();

		pt.updateScene( vertices, vcolors, vemissions, stream, origin, dps, gridRes );
		
		swUpdate.stop();

		OroStopwatch swRender( stream );
		swRender.start();

		pt.clearFrameBuffer( stream );

		for( int iteration = 0; iteration < 8; iteration++ )
		{
			pt.step( stream, camera, focus, lensR );
		}

		loadSceneFrame( frame ); // load next frame

		pt.resolve( stream );
		
		Buffer* frameBuffer = nullptr;
		while( frameBufferPool.try_pop( frameBuffer ) == false )
			;

		oroMemcpyDtoDAsync( (oroDeviceptr)frameBuffer->data(), (oroDeviceptr)pt.m_frameBufferU8->data(), (uint64_t)renderWidth * renderHeigt * sizeof( uchar4 ), stream );

		swRender.stop();

		taskGroup.addElements( 1 );
		threadPool.enqueueTask( [frame, frameBuffer, &taskGroup, device, renderWidth, renderHeigt, &frameBufferPool]() {
			thread_local oroCtx context = 0;
			if( context == 0 )
			{
				oroCtxCreate( &context, 0, device );
				oroCtxSetCurrent( context );
			}

			Image2DRGBA8 image;
			image.allocate( renderWidth, renderHeigt );
			oroMemcpyDtoH( image.data(), (oroDeviceptr)frameBuffer->data(), (uint64_t)renderWidth * renderHeigt * sizeof( uchar4 ) );

			frameBufferPool.push( frameBuffer );

			Stopwatch swSave;
			char output[256];
			sprintf( output, "render_%03d.png", frame );
			image.saveAsPngUncompressed( GetDataPath( output ).c_str() );
			taskGroup.doneElements( 1 );
		} );

		printf( "[frame %d] total( %.1f s ) / update %.3f / render %.3f\n", frame, sw.elapsed(), swUpdate.getMs(), swRender.getMs() );
	}
	taskGroup.waitForAllElementsToFinish();

	Buffer* frameBuffer = nullptr;
	while( frameBufferPool.try_pop( frameBuffer ) )
	{
		delete frameBuffer;
	}

	printf( "done %f\n", sw.elapsed() );

	pt.cleanUp();

	return 0;
}