#include "pr.hpp"
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
	SetDataDir( ExecutableDir() );

	if( oroInitialize( (oroApi)( ORO_API_HIP | ORO_API_CUDA ), 0 ) )
	{
		printf( "failed to init..\n" );
		return 0;
	}
	int deviceIdx = 2;
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

	const char* input = "rtcamp.abc";
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

	Image2DRGBA8 image;
	image.allocate( renderWidth, renderHeigt );

	glm::vec3 center = { -0.131793f, -1.40424f, -0.975714f };
	float boxWide = 11.0f;
	glm::vec3 origin = center - glm::vec3( boxWide * 0.5f );

	int fromRes = 64;
	int toRes = 4096;
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

		printf( "[frame %d] update %.3f / render %.3f\n", frame, swUpdate.getMs(), swRender.getMs() );

		pt.toImage( stream, &image );

		char output[256];
		sprintf( output, "render_%04d.png", frame );
		image.saveAsPng( GetDataPath( output ).c_str() );
	}

	pt.cleanUp();

	return 0;
}