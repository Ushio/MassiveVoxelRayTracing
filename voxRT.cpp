#include "morton.hpp"
#include "pr.hpp"
#include "voxelMeshWriter.hpp"
#include "voxelization.hpp"
#include "intersectorEmbree.hpp"
#include "IntersectorOctree.hpp"
#include <iostream>
#include <memory>
#include <set>

void trianglesFlattened( std::shared_ptr<pr::FScene> scene, std::vector<glm::vec3>* vertices )
{
	using namespace pr;
	vertices->clear();

	scene->visitPolyMesh( [&]( std::shared_ptr<const FPolyMeshEntity> polymesh )
						  {
        ColumnView<int32_t> faceCounts( polymesh->faceCounts() );
	    ColumnView<int32_t> indices( polymesh->faceIndices() );
	    ColumnView<glm::vec3> positions( polymesh->positions() );
	    for( int i = 0; i < faceCounts.count(); i++ )
	    {
		    PR_ASSERT( faceCounts[i] == 3 ); // no quad support now.
		    for( int j = 0; j < 3; ++j )
		    {
			    int index = indices[i * 3 + j];
			    vertices->push_back( positions[index] );
		    }
	    } } );
}
inline void drawVoxelsWire( const std::vector<uint64_t>& mortonVoxels, const glm::vec3& origin, float dps, glm::u8vec3 color )
{
	using namespace pr;

	PrimBegin( PrimitiveMode::Lines );
	for( auto morton : mortonVoxels )
	{
		glm::uvec3 c;
		decodeMortonCode_PEXT( morton, &c.x, &c.y, &c.z );
		glm::vec3 p = origin + glm::vec3( c.x, c.y, c.z ) * dps;

		uint32_t i0 = PrimVertex( p, color );
		uint32_t i1 = PrimVertex( p + glm::vec3( dps, 0, 0 ), color );
		uint32_t i2 = PrimVertex( p + glm::vec3( dps, 0, dps ), color );
		uint32_t i3 = PrimVertex( p + glm::vec3( 0, 0, dps ), color );
		uint32_t i4 = PrimVertex( p + glm::vec3( 0, dps, 0 ), color );
		uint32_t i5 = PrimVertex( p + glm::vec3( dps, dps, 0 ), color );
		uint32_t i6 = PrimVertex( p + glm::vec3( dps, dps, dps ), color );
		uint32_t i7 = PrimVertex( p + glm::vec3( 0, dps, dps ), color );

		PrimIndex( i0 );
		PrimIndex( i1 );
		PrimIndex( i1 );
		PrimIndex( i2 );
		PrimIndex( i2 );
		PrimIndex( i3 );
		PrimIndex( i3 );
		PrimIndex( i0 );

		PrimIndex( i4 );
		PrimIndex( i5 );
		PrimIndex( i5 );
		PrimIndex( i6 );
		PrimIndex( i6 );
		PrimIndex( i7 );
		PrimIndex( i7 );
		PrimIndex( i4 );

		PrimIndex( i0 );
		PrimIndex( i4 );
		PrimIndex( i1 );
		PrimIndex( i5 );
		PrimIndex( i2 );
		PrimIndex( i6 );
		PrimIndex( i3 );
		PrimIndex( i7 );
	}
	PrimEnd();
}

int main()
{
	using namespace pr;

	Config config;
	config.ScreenWidth = 1920;
	config.ScreenHeight = 1080;
	config.SwapInterval = 0;
	Initialize( config );

	Camera3D camera;
	camera.origin = { 4, 4, 4 };
	camera.lookat = { 0, 0, 0 };
	camera.zUp = false;

	const char* input = "bunny.obj";
	SetDataDir( ExecutableDir() );
	std::string errorMsg;
	std::shared_ptr<FScene> scene = ReadWavefrontObj( GetDataPath( input ), errorMsg );

	std::vector<glm::vec3> vertices;
	trianglesFlattened( scene, &vertices );

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
	bool buildAccelerationStructure = true;
	bool renderParallel = false;

	enum DAGBUILD
	{
		DAGBUILD_NO,
		DAGBUILD_REF,
	};
	int dagBuild = DAGBUILD_REF;

	enum INTERSECTOR
	{
		INTERSECTOR_OCTREE,
		INTERSECTOR_EMBREE,
	};
	pr::ITexture* bgTexture = 0;
	std::shared_ptr<IntersectorOctree> octreeVoxel( new IntersectorOctree() );
	std::shared_ptr<IntersectorEmbree> embreeVoxel( new IntersectorEmbree() );
	int intersector = INTERSECTOR_OCTREE;

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

		Stopwatch sw;

		glm::vec3 origin = bbox_lower;
		glm::vec3 bbox_size = bbox_upper - bbox_lower;
		float dps = glm::max( glm::max( bbox_size.x, bbox_size.y ), bbox_size.z ) / (float)gridRes;

		if( buildAccelerationStructure )
		{
			mortonVoxels.clear();
			for( int i = 0; i < vertices.size(); i += 3 )
			{
				glm::vec3 v0 = vertices[i];
				glm::vec3 v1 = vertices[i + 1];
				glm::vec3 v2 = vertices[i + 2];

				VTContext context( v0, v1, v2, sixSeparating, origin, dps, gridRes );
				glm::ivec2 xrange = context.xRangeInclusive();
				for( int x = xrange.x; x <= xrange.y; x++ )
				{
					glm::ivec2 yrange = context.yRangeInclusive( x, dps );
					for( int y = yrange.x; y <= yrange.y; y++ )
					{
						glm::ivec2 zrange = context.zRangeInclusive( x, y, dps, sixSeparating );
						for( int z = zrange.x; z <= zrange.y; z++ )
						{
							glm::vec3 p = context.p( x, y, z, dps );
							if( context.intersect( p ) )
							{
								glm::ivec3 c = context.i( x, y, z );
								mortonVoxels.push_back( encode2mortonCode_PDEP( c.x, c.y, c.z ) );
							}
						}
					}
				}
			}
		}

		double voxelizationTime = sw.elapsed();

		// mortonVoxels has some duplications but I don't care now.
		if( drawWire )
		{
			drawVoxelsWire( mortonVoxels, origin, dps, { 200, 200, 200 } );
		}

		double octreeBuildMS = 0.0;
		double embreeBuildMS = 0.0;
		if( buildAccelerationStructure )
		{
			static std::set<uint64_t> accInputs;
			accInputs.clear();
			for( auto v : mortonVoxels )
			{
				accInputs.insert( v );
			}

			sw = Stopwatch();
			embreeVoxel->build( accInputs, origin, dps );
			embreeBuildMS = sw.elapsed() * 1000.0;
			sw = Stopwatch();
			if( dagBuild == DAGBUILD_NO )
			{
				octreeVoxel->build( std::vector<uint64_t>( accInputs.begin(), accInputs.end() ), origin, dps, gridRes );
			}
			else if( dagBuild == DAGBUILD_REF )
			{
				octreeVoxel->buildDAGReference( std::vector<uint64_t>( accInputs.begin(), accInputs.end() ), origin, dps, gridRes );
			}
			octreeBuildMS = sw.elapsed() * 1000.0;
		}

		// A single ray test
		{
			static glm::vec3 from = { -3, -3, -3 };
			static glm::vec3 to = { -0.415414095, 1.55378413, 1.55378413 };
			ManipulatePosition( camera, &from, 1 );
			ManipulatePosition( camera, &to, 1 );

			DrawText( from, "from" );
			DrawText( to, "to" );
			DrawLine( from, to, { 128, 128, 128 } );

			float t = FLT_MAX;
			int nMajor;
			glm::vec3 ro = from;
			glm::vec3 rd = to - from;
			octreeVoxel->intersect( ro, rd, &t, &nMajor );

			DrawSphere( ro + rd * t, 0.01f, { 255, 0, 0 } );

			glm::vec3 hitN = unProjectPlane( { 0.0f, 0.0f }, project2plane_reminder( rd, nMajor ) < 0.0f ? 1.0f : -1.0f, nMajor );
			DrawArrow( ro + rd * t, ro + rd * t + hitN * 0.1f, 0.01f, { 255, 0, 0 } );
		}

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
				if( intersector == INTERSECTOR_EMBREE )
				{
					embreeVoxel->intersect( ro, rd, &t, &nMajor );
				}
				else if( intersector == INTERSECTOR_OCTREE )
				{
					octreeVoxel->intersect( ro, rd, &t, &nMajor );
				}

				if( t != FLT_MAX )
				{
					glm::vec3 hitN = unProjectPlane( { 0.0f, 0.0f }, project2plane_reminder( rd, nMajor ) < 0.0f ? 1.0f : -1.0f, nMajor );
					glm::vec3 color = ( hitN + glm::vec3( 1.0f ) ) * 0.5f;
					image( i, j ) = { 255 * color.r, 255 * color.g, 255 * color.b, 255 };
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
		ImGui::Text( "voxelization(ms) = %f", voxelizationTime * 1000.0 );
		ImGui::Text( "octree build(ms) = %f", octreeBuildMS );
		ImGui::Text( "embree build(ms) = %f", embreeBuildMS );
		ImGui::Text( "octree   = %lld byte", octreeVoxel->getMemoryConsumption() );
		ImGui::Text( "embree = %lld byte", embreeVoxel->getMemoryConsumption() );
		ImGui::Text( "RT (ms) = %f", RT_MS );
		ImGui::Checkbox( "renderParallel", &renderParallel );
		ImGui::RadioButton( "Intersector: Octree", &intersector, INTERSECTOR_OCTREE );
		ImGui::RadioButton( "Intersector: Embree", &intersector, INTERSECTOR_EMBREE );

		ImGui::RadioButton( "DAG: none", &dagBuild, DAGBUILD_NO );
		ImGui::RadioButton( "DAG: reference", &dagBuild, DAGBUILD_REF );
		
		ImGui::End();

		EndImGui();
	}

	pr::CleanUp();
}