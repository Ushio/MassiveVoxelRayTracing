#include "pr.hpp"
#include "voxelization.hpp"
#include "morton.hpp"
#include "voxelMeshWriter.hpp"
#include <iostream>
#include <memory>
#include <set>

#include "voxUtil.hpp"

inline void drawVoxelsFace(const std::vector<uint64_t>& mortonVoxels, const glm::vec3& origin, float dps )
{
	using namespace pr;

	TriBegin( 0 );
	for( auto morton : mortonVoxels )
	{
		glm::uvec3 c;
		decodeMortonCode_PEXT( morton, &c.x, &c.y, &c.z );
		glm::vec3 p = origin + glm::vec3( c.x, c.y, c.z ) * dps;
		float x = p.x;
		float y = p.y;
		float z = p.z;

#define F( a, b, c )        \
	TriIndex( indices[a] ); \
	TriIndex( indices[b] ); \
	TriIndex( indices[c] );

		// XZ
		{
			uint32_t indices[] = {
				TriVertex( { x, y + dps, z }, { 0.0f, 0.0f }, { 0, 255, 0, 255 } ),
				TriVertex( { x + dps, y + dps, z }, { 0.0f, 0.0f }, { 0, 255, 0, 255 } ),
				TriVertex( { x + dps, y + dps, z + dps }, { 0.0f, 0.0f }, { 0, 255, 0, 255 } ),
				TriVertex( { x, y + dps, z + dps }, { 0.0f, 0.0f }, { 0, 255, 0, 255 } ),
			};
			F( 0, 1, 2 );
			F( 2, 3, 0 );
		}
		{
			uint32_t indices[] = {
				TriVertex( { x, y, z }, { 0.0f, 0.0f }, { 0, 255, 0, 255 } ),
				TriVertex( { x + dps, y, z }, { 0.0f, 0.0f }, { 0, 255, 0, 255 } ),
				TriVertex( { x + dps, y, z + dps }, { 0.0f, 0.0f }, { 0, 255, 0, 255 } ),
				TriVertex( { x, y, z + dps }, { 0.0f, 0.0f }, { 0, 255, 0, 255 } ),
			};
			F( 0, 1, 2 );
			F( 2, 3, 0 );
		}

		// YZ
		{
			uint32_t indices[] = {
				TriVertex( { x + dps, y, z }, { 0.0f, 0.0f }, { 255, 0, 0, 255 } ),
				TriVertex( { x + dps, y, z + dps }, { 0.0f, 0.0f }, { 255, 0, 0, 255 } ),
				TriVertex( { x + dps, y + dps, z + dps }, { 0.0f, 0.0f }, { 255, 0, 0, 255 } ),
				TriVertex( { x + dps, y + dps, z }, { 0.0f, 0.0f }, { 255, 0, 0, 255 } ),
			};
			F( 0, 1, 2 );
			F( 2, 3, 0 );
		}
		{
			uint32_t indices[] = {
				TriVertex( { x, y, z }, { 0.0f, 0.0f }, { 255, 0, 0, 255 } ),
				TriVertex( { x, y, z + dps }, { 0.0f, 0.0f }, { 255, 0, 0, 255 } ),
				TriVertex( { x, y + dps, z + dps }, { 0.0f, 0.0f }, { 255, 0, 0, 255 } ),
				TriVertex( { x, y + dps, z }, { 0.0f, 0.0f }, { 255, 0, 0, 255 } ),
			};
			F( 0, 1, 2 );
			F( 2, 3, 0 );
		}
		// XY
		{
			uint32_t indices[] = {
				TriVertex( { x, y, z + dps }, { 0.0f, 0.0f }, { 0, 0, 255, 255 } ),
				TriVertex( { x, y + dps, z + dps }, { 0.0f, 0.0f }, { 0, 0, 255, 255 } ),
				TriVertex( { x + dps, y + dps, z + dps }, { 0.0f, 0.0f }, { 0, 0, 255, 255 } ),
				TriVertex( { x + dps, y, z + dps }, { 0.0f, 0.0f }, { 0, 0, 255, 255 } ),
			};
			F( 0, 1, 2 );
			F( 2, 3, 0 );
		}
		{
			uint32_t indices[] = {
				TriVertex( { x, y, z }, { 0.0f, 0.0f }, { 0, 0, 255, 255 } ),
				TriVertex( { x, y + dps, z }, { 0.0f, 0.0f }, { 0, 0, 255, 255 } ),
				TriVertex( { x + dps, y + dps, z }, { 0.0f, 0.0f }, { 0, 0, 255, 255 } ),
				TriVertex( { x + dps, y, z }, { 0.0f, 0.0f }, { 0, 0, 255, 255 } ),
			};
			F( 0, 1, 2 );
			F( 2, 3, 0 );
		}

#undef F
	}
	TriEnd();
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

	// const char* input = "bunny.obj";
	const char* input = "bunnyColor.abc";
	SetDataDir( ExecutableDir() );

	std::string errorMsg;

	AbcArchive ar;
	ar.open( GetDataPath( input ), errorMsg );
	std::shared_ptr<FScene> scene = ar.readFlat( 0, errorMsg );
	
	// std::shared_ptr<FScene> scene = ReadWavefrontObj( GetDataPath( input ), errorMsg );

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
	int gridRes = 128;
	bool drawModel = true;
	bool drawWire = true;
	bool drawFace = true;

	SetDepthTest( true );

	while( pr::NextFrame() == false )
	{
		if( IsImGuiUsingMouse() == false )
		{
			UpdateCameraBlenderLike( &camera );
		}
		ClearBackground( 0.1f, 0.1f, 0.1f, 1 );

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
				for (int j = 0; j < 3; j++)
				{
					pr::PrimIndex( indices[j] );
					pr::PrimIndex( indices[( j + 1 ) % 3] );
				}
			}
			PrimEnd();
		}

		static std::vector<uint64_t> mortonVoxels;
		mortonVoxels.clear();

		Stopwatch sw;

		glm::vec3 origin = bbox_lower;
		glm::vec3 bbox_size = bbox_upper - bbox_lower;
		float dps = glm::max( glm::max( bbox_size.x, bbox_size.y ), bbox_size.z ) / (float)gridRes;

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

		double voxelizationTime = sw.elapsed();

		// mortonVoxels has some duplications but I don't care now.
		if( drawWire )
		{
			drawVoxelsWire( mortonVoxels, origin, dps, { 200, 200, 200 } );
		}
		if( drawFace )
		{
			drawVoxelsFace( mortonVoxels, origin, dps );
		}

		PopGraphicState();
		EndCamera();

		BeginImGui();

		ImGui::SetNextWindowSize( { 500, 800 }, ImGuiCond_Once );
		ImGui::Begin( "Panel" );
		ImGui::Text( "fps = %f", GetFrameRate() );

		ImGui::SeparatorText( "Voxlizaiton" );
		ImGui::InputInt( "gridRes", &gridRes );
		ImGui::Checkbox( "sixSeparating", &sixSeparating );

		ImGui::SeparatorText( "Perf" );
		ImGui::Text( "voxelization(ms) = %f", voxelizationTime * 1000.0 );

		ImGui::SeparatorText( "Drawing" );
		ImGui::Checkbox( "drawModel", &drawModel );
		ImGui::Checkbox( "drawWire", &drawWire );
		ImGui::Checkbox( "drawFace", &drawFace );

		ImGui::SeparatorText( "Save" );
		if( ImGui::Button( "Save As Mesh" ) )
		{
			VoxelMeshWriter writer;
			std::set<uint64_t> voxels( mortonVoxels.begin(), mortonVoxels.end() );

			for (auto m : voxels)
			{
				uint32_t x, y, z;
				decodeMortonCode_PEXT( m, &x, &y, &z );
				writer.add( { origin.x + x * dps, origin.y + y * dps, origin.z + z * dps }, dps );
			}

			writer.savePLY( GetDataPath( "vox.ply" ).c_str() );
		}

		ImGui::End();

		EndImGui();
	}

	pr::CleanUp();
}