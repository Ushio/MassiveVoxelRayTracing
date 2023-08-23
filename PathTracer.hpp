#pragma once
#include <memory>
#include "Orochi/Orochi.h"
#include "Orochi/OrochiUtils.h"
#include "voxUtil.hpp"
#include "hipUtil.hpp"

#include "IntersectorOctreeGPU.hpp"
#include "voxCommon.hpp"
#include "renderCommon.hpp"
#include "pmjSampler.hpp"
#include "StreamCompaction.hpp"

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

	void loadHDRI( oroStream stream, const char* file, const char* filePrimary = 0 )
	{
		pr::Image2DRGBA32 hdriSrc;
		hdriSrc.loadFromHDR( file );
		m_hdri.load( glm::value_ptr( *hdriSrc.data() ), hdriSrc.width(), hdriSrc.height(), m_voxKernel.get(), stream );

		if( filePrimary )
		{
			pr::Image2DRGBA32 hdriPrimarySrc;
			hdriPrimarySrc.loadFromHDR( filePrimary );
			m_hdri.loadPrimary( glm::value_ptr( *hdriPrimarySrc.data() ), hdriPrimarySrc.width(), hdriPrimarySrc.height(), stream );
		}
	}

	void toImageAsync( oroStream stream, pr::Image2DRGBA8* output )
	{
		output->allocate( m_width, m_height );

		ShaderArgument args;
		args.add( m_frameBufferU8->data() );
		args.add( m_frameBufferF32->data() );
		args.add( m_width * m_height );
		m_voxKernel->launch( "renderResolve", args, div_round_up64( m_width * m_height, 128 ), 1, 1, 128, 1, 1, stream );

		oroMemcpyDtoHAsync( output->data(), (oroDeviceptr)m_frameBufferU8->data(), (uint64_t)m_width * m_height * sizeof( uchar4 ), stream );
	}
	void resolve(oroStream stream)
	{
		ShaderArgument args;
		args.add( m_frameBufferU8->data() );
		args.add( m_frameBufferF32->data() );
		args.add( m_width * m_height );
		m_voxKernel->launch( "renderResolve", args, div_round_up64( m_width * m_height, 128 ), 1, 1, 128, 1, 1, stream );
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

	void step( oroStream stream, pr::Camera3D camera, float focus, float lensR )
	{
		glm::mat4 viewMat, projMat;
		GetCameraMatrix( camera, &projMat, &viewMat, m_width, m_height );

		CameraPinhole pinhole;
		pinhole.initFromPerspective( viewMat, projMat, focus, lensR );

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
