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

	printf( "Expected arguments:\n" );
	printf( "  [instance 0] RTCamp --frame-range 0 171\n" );
	printf( "  [instance 1] RTCamp --frame-range 171 240\n" );
	
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
	if( ar.open( GetDataPath( input ), errorMsg ) == AbcArchive::Result::Failure )
	{
		printf( "can't open scene file %s. Reason:\n", input );
		printf( "%s\n", errorMsg.c_str() );
		return 1;
	}

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
	pt.loadHDRI( stream, "monks_forest_s.hdr", "monks_forest_2k_primary.hdr" );

	// reading buffer
	Concurrency::concurrent_queue<Buffer*> frameBufferPool;
	for( int i = 0; i < 4; i++ )
	{
		frameBufferPool.push( new Buffer( (uint64_t)renderWidth * renderHeigt * sizeof( uchar4 ) ) );
	}

	glm::vec3 center = { 3.16434f, -1.40424f, -3.77277f };
	float boxWide = 20.0f;
	glm::vec3 origin = center - glm::vec3( boxWide * 0.5f );

	int fromRes = 256;
	int toRes = 8192;
	for( int frame = beginFrame; frame < endFrame; frame++ )
	{
		float dps = glm::mix( boxWide / fromRes, boxWide / toRes, (float)frame / totalFrames );
		int resolution = (int)ceil( boxWide / dps );
		int gridRes = next_power_of_two( resolution );

		OroStopwatch swUpdate( stream );
		swUpdate.start();

		pt.updateScene( vertices, vcolors, vemissions, stream, origin, dps, gridRes );
		
		swUpdate.stop();


		// CPU
		#if 1
		{
			IntersectorOctreeGPU intersector = pt.m_intersectorOctreeGPU.copyToHost();
			HDRI hdri = pt.m_hdri.copyToHost();
			PMJSampler pmj;
			pmj.setup( false, 0 );
			glm::mat4 viewMat, projMat;
			GetCameraMatrix( camera, &projMat, &viewMat, renderWidth, renderHeigt );

			CameraPinhole pinhole;
			pinhole.initFromPerspective( viewMat, projMat, focus, lensR );

			int nSPP = 16 * 8;
			Image2DRGBA8 image;
			image.allocate( renderWidth, renderHeigt );
			Stopwatch cpuRenderSw;

			//for( int y = 0; y < renderHeigt; y++ )
			ParallelFor( renderHeigt, [&]( int y ) {
			for (int x = 0; x < renderWidth; x++)
			{
				float3 contribution = {};
				for( uint32_t spp = 0; spp < nSPP; spp++ )
				{
					uint32_t pixelIdx = y * renderWidth + x;
					int2 resolution = { renderWidth, renderHeigt };
					MurmurHash32 hash( 0 );
					hash.combine( pixelIdx );
					StackElement stack[32];

	#if defined( USE_PMJ )
					int dim = 0;
					uint32_t stream = hash.getHash();
	#define SAMPLE_2D() pmj.sample2d( spp, dim++, stream )
	#else
					hash.combine( spp );
					PCG32 rng;
					rng.setup( 0, hash.getHash() );
	#define SAMPLE_2D() float2{ uniformf( rng.nextU32() ), uniformf( rng.nextU32() ) }
	#endif

					float2 cam_u01 = SAMPLE_2D();
					float3 ro, rd;
					// pinhole.shoot( &ro, &rd, x, y, cam_u01.x, cam_u01.y, resolution.x, resolution.y );

					float2 lens_u01 = SAMPLE_2D();
					pinhole.shootThinLens( &ro, &rd, x, y, cam_u01.x, cam_u01.y, resolution.x, resolution.y, lens_u01.x, lens_u01.y );

					float3 T = { 1.0f, 1.0f, 1.0f };
					float3 L = {};

					float t = MAX_FLOAT;
					int nMajor;
					uint32_t vIndex = 0;
					intersector.intersect( stack, ro, rd, &t, &nMajor, &vIndex, false /* isShadowRay */ );

					// Primary emissions:
					if( t == MAX_FLOAT )
					{
						// float I = ss_max( normalize( rd ).y, 0.0f ) * 3.0f;
						// float3 env = { I, I, I };
						float3 env = hdri.sampleNearest( rd, true );
						L += T * env;
					}
					else
					{
						float3 Le = intersector.getVoxelEmission( vIndex, false );
						L += T * Le;
					}

					for( int depth = 0; depth < 8 && t != MAX_FLOAT; depth++ )
					{
						float3 R = rawReflectance( intersector.getVoxelColor( vIndex ) );
						float3 hitN = getHitN( nMajor, rd );
						float3 hitP = ro + rd * t;

						if( hdri.isEnabled() )
						{ // Explicit
							float2 u01 = SAMPLE_2D();
							float2 u23 = SAMPLE_2D();

							float3 dir;
							float3 emissive;
							float p;
							hdri.importanceSample( &dir, &emissive, &p, hitN, true, u01.x, u01.y, u23.x, u23.y );

							// no self intersection
							float t = MAX_FLOAT;
							int nMajor;
							uint32_t vIndex = 0;
							intersector.intersect( stack, hitP, dir, &t, &nMajor, &vIndex, true /* isShadowRay */ );
							if( t == MAX_FLOAT )
							{
								L += T * ( R / PI ) * ss_max( dot( hitN, dir ), 0.0f ) * emissive / p;
							}
						}

						T *= R;

	#if defined( EXTRA_IMPLICIT_SAMPLING )
						int nSampleExtraDirect = intersector.hasEmission() ? 1 : 0;
						for( int k = 0; depth == 0 && k < nSampleExtraDirect; k++ )
						{
							float2 u01 = SAMPLE_2D();

							float3 dir = sampleLambertian( u01.x, u01.y, hitN );

							// no self intersection
							float t = MAX_FLOAT;
							int nMajor;
							uint32_t vIndex = 0;
							intersector.intersect( stack, hitP, dir, &t, &nMajor, &vIndex, false /* isShadowRay */ );
							float3 Le = intersector.getVoxelEmission( vIndex, true );
							if( t != MAX_FLOAT )
							{
								L += T * Le / (float)( 1 + nSampleExtraDirect );
							}
						}
	#endif

						float2 u01 = SAMPLE_2D();
						float3 dir = sampleLambertian( u01.x, u01.y, hitN );

						ro = hitP; // no self intersection
						rd = dir;

						t = MAX_FLOAT;
						intersector.intersect( stack, ro, rd, &t, &nMajor, &vIndex, false /* isShadowRay */ );

						if( t != MAX_FLOAT )
						{
							float3 Le = intersector.getVoxelEmission( vIndex, true );

	#if defined( EXTRA_IMPLICIT_SAMPLING )
							L += T * Le * ( depth == 0 ? 1.0f / (float)( 1 + nSampleExtraDirect ) : 1.0f );
	#else
							L += T * Le;
	#endif
						}
					}

					contribution += L;
				}
				float w = nSPP;
				int r = (int)( 255 * INTRIN_POW( contribution.x / w, 1.0f / 2.2f ) + 0.5f );
				int g = (int)( 255 * INTRIN_POW( contribution.y / w, 1.0f / 2.2f ) + 0.5f );
				int b = (int)( 255 * INTRIN_POW( contribution.z / w, 1.0f / 2.2f ) + 0.5f );
				glm::u8vec4 colorOut = {
					(uint8_t)ss_min( r, 255 ),
					(uint8_t)ss_min( g, 255 ),
					(uint8_t)ss_min( b, 255 ),
					255 };
				image( x, y ) = colorOut;
			}
			} );

			printf( "CPU %fs\n", cpuRenderSw.elapsed() );
			image.saveAsPng( "test.png" );
		}
		#endif

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
			sprintf( output, "%03d.png", frame );
			image.saveAsPngUncompressed( GetDataPath( output ).c_str() );
			taskGroup.doneElements( 1 );
		} );

		printf( "[frame %d] res( %d ) total( %.1f s ) / update %.3f / render %.3f\n", frame, resolution, sw.elapsed(), swUpdate.getMs(), swRender.getMs() );
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