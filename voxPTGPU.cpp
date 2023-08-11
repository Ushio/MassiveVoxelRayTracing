#include "pr.hpp"
#include <iostream>
#include <memory>
#include <set>

#include "Orochi/Orochi.h"
#include "Orochi/OrochiUtils.h"
#include "voxUtil.hpp"
#include "hipUtil.hpp"

#include "IntersectorOctreeGPU.hpp"
#include "voxCommon.hpp"
#include "renderCommon.hpp"
#include "pmjSampler.hpp"
#include "StreamCompaction.hpp"

#define RENDER_NUMBER_OF_THREAD 64

struct PathTracer
{
public:
	std::unique_ptr<Shader> m_voxKernel;
	IntersectorOctreeGPU m_intersectorOctreeGPU;
	DynamicAllocatorGPU<StackElement> m_stackAllocator;
	PMJSampler m_pmj;
	HDRI m_hdri;

	std::unique_ptr<Buffer> m_frameBufferU8;
	std::unique_ptr<Buffer> m_frameBufferF32;
	int m_width = 0;
	int m_height = 0;
	int m_steps = 0;

	Shader& shader()
	{
		return *m_voxKernel;
	}
	int getSteps() const { return m_steps; }
	uint64_t getNumberOfVoxels() const
	{
		return m_intersectorOctreeGPU.m_numberOfVoxels;
	}
	uint64_t getOctreeBytes() const
	{
		return (uint64_t)m_intersectorOctreeGPU.m_numberOfNodes * sizeof( OctreeNode );
	}

	void setup( oroStream stream, const char* kernel, const char* includeDir, bool isNvidia )
	{
		m_pmj.setup( true, stream );
		m_stackAllocator.setup( 16 * 256 /* numberOfBlock */, RENDER_NUMBER_OF_THREAD /* blockSize */, isNvidia ? 32 : 37 /* nElementPerThread */, stream );

		std::vector<char> voxSrc;
		loadFileAsVector( &voxSrc, kernel );
		voxSrc.push_back( '\0' );

		std::vector<std::string> compilerArgs;
		compilerArgs.push_back( "-I" + std::string( includeDir ) );

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

		m_voxKernel = std::unique_ptr<Shader>( new Shader( voxSrc.data(), "voxKernel.cu", compilerArgs ) );
	}

	void cleanUp()
	{
		m_intersectorOctreeGPU.cleanUp();
		m_stackAllocator.cleanUp();
		m_pmj.cleanUp( true );
		m_voxKernel = std::unique_ptr<Shader>();
		m_frameBufferU8 = std::unique_ptr<Buffer>();
		m_frameBufferF32 = std::unique_ptr<Buffer>();
	}

	void resizeFrameBufferIfNeeded( oroStream stream, int width, int height )
	{
		uint64_t frameBufferF32Bytes = (uint64_t)width * height * sizeof( float4 );
		if( !m_frameBufferF32 || m_frameBufferF32->bytes() != frameBufferF32Bytes )
		{
			m_frameBufferF32 = std::unique_ptr<Buffer>( new Buffer( frameBufferF32Bytes ) );
			clearFrameBuffer( stream );
		}
		uint64_t frameBufferU8Bytes = (uint64_t)width * height * sizeof( uchar4 );
		if( !m_frameBufferU8 || m_frameBufferU8->bytes() != frameBufferU8Bytes )
		{
			m_frameBufferU8 = std::unique_ptr<Buffer>( new Buffer( frameBufferU8Bytes ) );
		}

		m_width = width;
		m_height = height;
	}
	void clearFrameBuffer( oroStream stream )
	{
		m_steps = 0;
		oroMemsetD32Async( (oroDeviceptr)m_frameBufferF32->data(), 0, m_frameBufferF32->bytes() / 4, stream );
	}

	void loadHDRI( oroStream stream, const char* file )
	{
		pr::Image2DRGBA32 hdriSrc;
		hdriSrc.loadFromHDR( file );
		m_hdri.load( glm::value_ptr( *hdriSrc.data() ), hdriSrc.width(), hdriSrc.height(), m_voxKernel.get(), stream );
	}

	void toImage( oroStream stream, pr::Image2DRGBA8* output )
	{
		output->allocate( m_width, m_height );

		ShaderArgument args;
		args.add( m_frameBufferU8->data() );
		args.add( m_frameBufferF32->data() );
		args.add( m_width * m_height );
		m_voxKernel->launch( "renderResolve", args, div_round_up64( m_width * m_height, 128 ), 1, 1, 128, 1, 1, stream );
		
		oroMemcpyDtoHAsync( output->data(), (oroDeviceptr)m_frameBufferU8->data(), (uint64_t)m_width * m_height * sizeof( uchar4 ), stream );
		oroStreamSynchronize( stream );
	}

	void updateScene( const std::vector<glm::vec3>& vertices,
					  const std::vector<glm::vec3>& vcolors,
					  const std::vector<glm::vec3>& vemissions,
					  oroStream stream,
					  glm::vec3 origin,
					  float dps,
					  int gridRes )
	{
		m_intersectorOctreeGPU.build( vertices, vcolors, vemissions, m_voxKernel.get(), stream, origin, dps, gridRes );
	}

	void step( oroStream stream, pr::Camera3D camera )
	{
		glm::mat4 viewMat, projMat;
		GetCameraMatrix( camera, &projMat, &viewMat, m_width, m_height );

		CameraPinhole pinhole;
		pinhole.initFromPerspective( viewMat, projMat );

		ShaderArgument args;
		args.add( m_steps++ );
		args.add( m_frameBufferF32->data() );
		args.add<int2>( { m_width, m_height } );
		args.add( pinhole );
		args.add( m_intersectorOctreeGPU );
		args.add( m_stackAllocator );
		args.add( m_hdri );
		args.add( m_pmj );

		m_voxKernel->launch( "renderPT", args, div_round_up64( m_width * m_height, RENDER_NUMBER_OF_THREAD ), 1, 1, RENDER_NUMBER_OF_THREAD, 1, 1, stream );
	}
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

			pt.toImage( stream, &image );

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