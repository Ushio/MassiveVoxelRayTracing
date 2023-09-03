#include "morton.hpp"
#include "pr.hpp"
#include "voxelMeshWriter.hpp"
#include "voxelization.hpp"
#include "intersectorEmbree.hpp"
#include "IntersectorOctree.hpp"
#include <iostream>
#include <memory>
#include <set>

#include "voxUtil.hpp"
#include "renderCommon.hpp"

void mergeVoxels( std::vector<uint64_t>* keys, std::vector<glm::u8vec4>* values, std::vector<glm::u8vec4>* emissions )
{
	struct Attribute
	{
		glm::uvec4 color;
		glm::uvec4 emission;
	};
	std::map<uint64_t, Attribute> voxels;
	for (int i = 0; i < keys->size(); i++)
	{
		auto key = ( *keys )[i];
		auto value = ( *values )[i];
		auto emission = ( *emissions )[i];
		auto it = voxels.find( key );
		if( it == voxels.end() )
		{
			voxels[key] = { { value.x, value.y, value.z, 1 }, { emission.x, emission.y, emission.z, 1 } };
		}
		else
		{
			it->second.color += glm::vec4{ value.x, value.y, value.z, 1 };
			it->second.emission += glm::vec4{ emission.x, emission.y, emission.z, 1 };
		}
	}

	keys->clear();
	values->clear();
	emissions->clear();

	for( auto kv : voxels )
	{
		keys->push_back( kv.first );
		auto v = glm::u8vec4( kv.second.color / kv.second.color.w );
		values->push_back( v );
		auto e = glm::u8vec4( kv.second.emission / kv.second.emission.w );
		emissions->push_back( e );
	}
}

inline float3 toFloat3( glm::vec3 v )
{
	return { v.x, v.y, v.z };
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

	SetDataDir( ExecutableDir() );

	const char* input = "xyzrgb_dragon.abc";
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
	std::vector<glm::vec3> vemissions;
	trianglesFlattened( scene, &vertices, &vcolors, &vemissions );

	glm::vec3 bbox_lower = glm::vec3( FLT_MAX );
	glm::vec3 bbox_upper = glm::vec3( -FLT_MAX );
	for( int i = 0; i < vertices.size(); i++ )
	{
		bbox_lower = glm::min( bbox_lower, vertices[i] );
		bbox_upper = glm::max( bbox_upper, vertices[i] );
	}

	bool sixSeparating = true;
	int gridRes = 512;
	bool drawModel = false;
	bool drawWire = false;

	enum VIEWMODE
	{
		VIEWMODE_COLOR,
		VIEWMODE_EMISSION,
		VIEWMODE_NORMAL,
	};
	int viewMode = VIEWMODE_COLOR;

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
		static std::vector<glm::u8vec4> voxelColors;
		static std::vector<glm::u8vec4> voxelEmissions;
		Stopwatch sw;

		glm::vec3 origin = bbox_lower;
		glm::vec3 bbox_size = bbox_upper - bbox_lower;
		float dps = glm::max( glm::max( bbox_size.x, bbox_size.y ), bbox_size.z ) / (float)gridRes;

		if( buildAccelerationStructure )
		{
			mortonVoxels.clear();
			voxelColors.clear();
			voxelEmissions.clear();

			for( int i = 0; i < vertices.size(); i += 3 )
			{
				float3 v0 = toFloat3( vertices[i] );
				float3 v1 = toFloat3( vertices[i + 1] );
				float3 v2 = toFloat3( vertices[i + 2] );

				float3 c0 = toFloat3( vcolors[i] );
				float3 c1 = toFloat3( vcolors[i + 1] );
				float3 c2 = toFloat3( vcolors[i + 2] );

				float3 e0 = toFloat3( vemissions[i] );
				float3 e1 = toFloat3( vemissions[i + 1] );
				float3 e2 = toFloat3( vemissions[i + 2] );

				VTContext context( { v0.x, v0.y, v0.z }, { v1.x, v1.y, v1.z }, { v2.x, v2.y, v2.z }, sixSeparating, { origin.x, origin.y, origin.z }, dps, gridRes );
				int2 xrange = context.xRangeInclusive();
				for( int x = xrange.x; x <= xrange.y; x++ )
				{
					int2 yrange = context.yRangeInclusive( x, dps );
					for( int y = yrange.x; y <= yrange.y; y++ )
					{
						int2 zrange = context.zRangeInclusive( x, y, dps, sixSeparating );
						for( int z = zrange.x; z <= zrange.y; z++ )
						{
							float3 p = context.p( x, y, z, dps );
							if( context.intersect( p ) )
							{
								int3 c = context.i( x, y, z );
								mortonVoxels.push_back( encode2mortonCode_PDEP( c.x, c.y, c.z ) );

								float3 bc = closestBarycentricCoordinateOnTriangle( v0, v1, v2, p );
								float3 bColor = bc.x * c1 + bc.y * c2 + bc.z * c0;
								glm::u8vec4 voxelColor = { bColor.x * 255.0f + 0.5f, bColor.y * 255.0f + 0.5f, bColor.z * 255.0f + 0.5f, 255 };
								voxelColors.push_back( voxelColor );

								float3 bEmission = bc.x * e1 + bc.y * e2 + bc.z * e0;
								glm::u8vec4 voxelEmission = { bEmission.x * 255.0f + 0.5f, bEmission.y * 255.0f + 0.5f, bEmission.z * 255.0f + 0.5f, 255 };
								voxelEmissions.push_back( voxelEmission );
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
			mergeVoxels( &mortonVoxels, &voxelColors, &voxelEmissions );

			sw = Stopwatch();
			embreeVoxel->build( mortonVoxels, origin, dps );
			embreeBuildMS = sw.elapsed() * 1000.0;
			sw = Stopwatch();
			if( dagBuild == DAGBUILD_NO )
			{
				octreeVoxel->build( mortonVoxels, origin, dps, gridRes );
			}
			else if( dagBuild == DAGBUILD_REF )
			{
				octreeVoxel->buildDAGReference( mortonVoxels, origin, dps, gridRes );
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
			uint32_t vIndex;
			glm::vec3 ro = from;
			glm::vec3 rd = to - from;
			octreeVoxel->intersect( { ro.x, ro.y, ro.z }, { rd.x, rd.y, rd.z }, &t, &nMajor, &vIndex );

			DrawSphere( ro + rd * t, 0.01f, { 255, 0, 0 } );

			glm::vec3 hitN = getHitN( nMajor, rd );
			DrawArrow( ro + rd * t, ro + rd * t + hitN * 0.1f, 0.01f, { 255, 0, 0 } );
		}

		#if 1
		Image2DRGBA8 image;
		image.allocate( GetScreenWidth(), GetScreenHeight() );

		CameraPinhole pinhole;
		pinhole.initFromPerspective( GetCurrentViewMatrix(), GetCurrentProjMatrix(), 1.0f, 0.0f );

		//CameraRayGenerator rayGenerator( GetCurrentViewMatrix(), GetCurrentProjMatrix(), image.width(), image.height() );

		sw = Stopwatch();

		auto renderLine = [&]( int j ) {
			for( int i = 0; i < image.width(); ++i )
			{
				float3 ro, rd;
				pinhole.shoot( &ro, &rd, i, j, 0.5f, 0.5f, image.width(), image.height() );

				float t = FLT_MAX;
				int nMajor;
				uint32_t vIndex = 0;
				if( intersector == INTERSECTOR_EMBREE )
				{
					embreeVoxel->intersect( ro, rd, &t, &nMajor );
				}
				else if( intersector == INTERSECTOR_OCTREE )
				{
					octreeVoxel->intersect( ro, rd, &t, &nMajor, &vIndex );
				}

				if( t != FLT_MAX )
				{
					float3 hitN = getHitN( nMajor, rd );
					switch( viewMode )
					{
					case VIEWMODE_COLOR:
						image( i, j ) = voxelColors[vIndex];
						break;
					case VIEWMODE_EMISSION:
						image( i, j ) = voxelEmissions[vIndex];
						break;
					case VIEWMODE_NORMAL:
						float3 color = ( hitN + float3{ 1.0f, 1.0f, 1.0f } ) * 0.5f;
						image( i, j ) = { 255 * color.x + 0.5f, 255 * color.y + 0.5f, 255 * color.z + 0.5f, 255 };
						break;
					}
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
		double RT_S = sw.elapsed();

		if( bgTexture == nullptr )
		{
			bgTexture = CreateTexture();
		}
		bgTexture->upload( image );
#else
		double RT_MS = 0;
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
		ImGui::RadioButton( "viewMode: color", &viewMode, VIEWMODE_COLOR );
		ImGui::RadioButton( "viewMode: emission", &viewMode, VIEWMODE_EMISSION );
		ImGui::RadioButton( "viewMode: normal", &viewMode, VIEWMODE_NORMAL );

		ImGui::Text( "voxelization(ms) = %f", voxelizationTime * 1000.0 );
		ImGui::Text( "octree build(ms) = %f", octreeBuildMS );
		ImGui::Text( "embree build(ms) = %f", embreeBuildMS );
		ImGui::Text( "octree   = %lld byte", octreeVoxel->getMemoryConsumption() );
		ImGui::Text( "embree = %lld byte", embreeVoxel->getMemoryConsumption() );
		ImGui::Text( "voxels = %lld", mortonVoxels.size() );

		ImGui::Text( "RT (s) = %f", RT_S );
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