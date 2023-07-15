#include "pr.hpp"
#include "voxelization.hpp"
#include "morton.hpp"
#include "voxelMeshWriter.hpp"
#include <iostream>
#include <memory>
#include <set>

#include "voxUtil.hpp"
#include "voxCommon.hpp"

inline void drawVoxelsFace( const std::vector<uint64_t>& mortonVoxels, const std::vector<glm::u8vec4>& colors, const glm::vec3& origin, float dps )
{
	using namespace pr;

	TriBegin( 0 );
	for( int i = 0; i < mortonVoxels.size() ; i++ )
	{
		auto morton = mortonVoxels[i];
		glm::u8vec4 R = !colors.empty() ? colors[i] : glm::u8vec4{ 255, 0, 0, 255 };
		glm::u8vec4 G = !colors.empty() ? colors[i] : glm::u8vec4{ 0, 255, 0, 255 };
		glm::u8vec4 B = !colors.empty() ? colors[i] : glm::u8vec4{ 0, 0, 255, 255 };

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
				TriVertex( { x, y + dps, z }, { 0.0f, 0.0f }, G ),
				TriVertex( { x + dps, y + dps, z }, { 0.0f, 0.0f }, G ),
				TriVertex( { x + dps, y + dps, z + dps }, { 0.0f, 0.0f }, G ),
				TriVertex( { x, y + dps, z + dps }, { 0.0f, 0.0f }, G ),
			};
			F( 0, 1, 2 );
			F( 2, 3, 0 );
		}
		{
			uint32_t indices[] = {
				TriVertex( { x, y, z }, { 0.0f, 0.0f }, G ),
				TriVertex( { x + dps, y, z }, { 0.0f, 0.0f }, G ),
				TriVertex( { x + dps, y, z + dps }, { 0.0f, 0.0f }, G ),
				TriVertex( { x, y, z + dps }, { 0.0f, 0.0f }, G ),
			};
			F( 0, 1, 2 );
			F( 2, 3, 0 );
		}

		// YZ
		{
			uint32_t indices[] = {
				TriVertex( { x + dps, y, z }, { 0.0f, 0.0f }, R ),
				TriVertex( { x + dps, y, z + dps }, { 0.0f, 0.0f }, R ),
				TriVertex( { x + dps, y + dps, z + dps }, { 0.0f, 0.0f }, R ),
				TriVertex( { x + dps, y + dps, z }, { 0.0f, 0.0f }, R ),
			};
			F( 0, 1, 2 );
			F( 2, 3, 0 );
		}
		{
			uint32_t indices[] = {
				TriVertex( { x, y, z }, { 0.0f, 0.0f }, R ),
				TriVertex( { x, y, z + dps }, { 0.0f, 0.0f }, R ),
				TriVertex( { x, y + dps, z + dps }, { 0.0f, 0.0f }, R ),
				TriVertex( { x, y + dps, z }, { 0.0f, 0.0f }, R ),
			};
			F( 0, 1, 2 );
			F( 2, 3, 0 );
		}
		// XY
		{
			uint32_t indices[] = {
				TriVertex( { x, y, z + dps }, { 0.0f, 0.0f }, B ),
				TriVertex( { x, y + dps, z + dps }, { 0.0f, 0.0f }, B ),
				TriVertex( { x + dps, y + dps, z + dps }, { 0.0f, 0.0f }, B ),
				TriVertex( { x + dps, y, z + dps }, { 0.0f, 0.0f }, B ),
			};
			F( 0, 1, 2 );
			F( 2, 3, 0 );
		}
		{
			uint32_t indices[] = {
				TriVertex( { x, y, z }, { 0.0f, 0.0f }, B ),
				TriVertex( { x, y + dps, z }, { 0.0f, 0.0f }, B ),
				TriVertex( { x + dps, y + dps, z }, { 0.0f, 0.0f }, B ),
				TriVertex( { x + dps, y, z }, { 0.0f, 0.0f }, B ),
			};
			F( 0, 1, 2 );
			F( 2, 3, 0 );
		}

#undef F
	}
	TriEnd();
}

inline float3 toFloat3( glm::vec3 v )
{
	return { v.x, v.y, v.z };
}

inline void saveVoxelsAsMesh( const char *file, const std::vector<uint64_t>& sortedMortons, glm::vec3 origin, float dps )
{
	std::vector<glm::vec3> points;
	points.reserve( sortedMortons.size() * 8 );
	for( uint64_t morton : sortedMortons )
	{
		uint32_t x, y, z;
		decodeMortonCode_PEXT( morton, &x, &y, &z );
		glm::vec3 p = { origin.x + x * dps, origin.y + y * dps, origin.z + z * dps };

		points.push_back( p );
		points.push_back( p + glm::vec3( dps, 0, 0 ) );
		points.push_back( p + glm::vec3( dps, 0, dps ) );
		points.push_back( p + glm::vec3( 0, 0, dps ) );
		points.push_back( p + glm::vec3( 0, dps, 0 ) );
		points.push_back( p + glm::vec3( dps, dps, 0 ) );
		points.push_back( p + glm::vec3( dps, dps, dps ) );
		points.push_back( p + glm::vec3( 0, dps, dps ) );
	}

	FILE* fp = fopen( file, "wb" );

	int nVoxels = points.size() / 8;
	std::vector<uint8_t> bytes;
	bytes.reserve( ( 1 + sizeof( uint32_t ) * 4 ) * 6 * nVoxels );
	uint32_t head = 0;
	uint32_t nface = 0;
	for( int i = 0; i < nVoxels; i++ )
	{
		uint32_t x, y, z;
		decodeMortonCode_PEXT( sortedMortons[i], &x, &y, &z );

		bool noXp = bSearch( sortedMortons.data(), sortedMortons.size(), encode2mortonCode_PDEP( x + 1, y, z ) ) == -1;
		bool noXm = x == 0 || bSearch( sortedMortons.data(), sortedMortons.size(), encode2mortonCode_PDEP( x - 1, y, z ) ) == -1;
		bool noYp = bSearch( sortedMortons.data(), sortedMortons.size(), encode2mortonCode_PDEP( x, y + 1, z ) ) == -1;
		bool noYm = y == 0 || bSearch( sortedMortons.data(), sortedMortons.size(), encode2mortonCode_PDEP( x, y - 1, z ) ) == -1;
		bool noZp = bSearch( sortedMortons.data(), sortedMortons.size(), encode2mortonCode_PDEP( x, y, z + 1 ) ) == -1;
		bool noZm = z == 0 || bSearch( sortedMortons.data(), sortedMortons.size(), encode2mortonCode_PDEP( x, y, z - 1 ) ) == -1;

		uint32_t i0 = i * 8;
		uint32_t i1 = i * 8 + 1;
		uint32_t i2 = i * 8 + 2;
		uint32_t i3 = i * 8 + 3;
		uint32_t i4 = i * 8 + 4;
		uint32_t i5 = i * 8 + 5;
		uint32_t i6 = i * 8 + 6;
		uint32_t i7 = i * 8 + 7;

#define F( a, b, c, d )                             \
	bytes.resize( bytes.size() + sizeof( uint32_t ) * 4 + 1 ); nface++; \
	bytes[head++] = 4;                              \
	memcpy( &bytes[head], &a, sizeof( uint32_t ) ); \
	head += 4;                                      \
	memcpy( &bytes[head], &b, sizeof( uint32_t ) ); \
	head += 4;                                      \
	memcpy( &bytes[head], &c, sizeof( uint32_t ) ); \
	head += 4;                                      \
	memcpy( &bytes[head], &d, sizeof( uint32_t ) ); \
	head += 4;

		// Left Hand
		if( noYm )
		{
			F( i3, i2, i1, i0 );
		}

		if( noYp )
		{
			F( i4, i5, i6, i7 );
		}

		if( noZm )
		{
			F( i0, i1, i5, i4 );
		}

		if( noXp )
		{
			F( i1, i2, i6, i5 );
		}

		if( noZp )
		{
			F( i2, i3, i7, i6 );
		}

		if( noXm )
		{
			F( i3, i0, i4, i7 );
		}
#undef F
	}

	// PLY header
	fprintf( fp, "ply\n" );
	fprintf( fp, "format binary_little_endian 1.0\n" );
	fprintf( fp, "element vertex %llu\n", points.size() );
	fprintf( fp, "property float x\n" );
	fprintf( fp, "property float y\n" );
	fprintf( fp, "property float z\n" );
	fprintf( fp, "element face %d\n", nface );
	fprintf( fp, "property list uchar uint vertex_indices\n" );
	fprintf( fp, "end_header\n" );

	// Write vertices
	fwrite( points.data(), sizeof( glm::vec3 ) * points.size(), 1, fp );
	fwrite( bytes.data(), bytes.size(), 1, fp );
	fclose( fp );
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
	// const char* input = "color.abc";
	SetDataDir( ExecutableDir() );

	std::string errorMsg;

	AbcArchive ar;
	ar.open( GetDataPath( input ), errorMsg );
	std::shared_ptr<FScene> scene = ar.readFlat( 0, errorMsg );
	
	// std::shared_ptr<FScene> scene = ReadWavefrontObj( GetDataPath( input ), errorMsg );

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
	int gridRes = 128;
	bool drawModel = true;
	bool drawWire = true;
	bool drawFace = true;

	bool showVertexColor = true;

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
		static std::vector<glm::u8vec4> voxelColors;
		mortonVoxels.clear();
		voxelColors.clear();

		Stopwatch sw;

		glm::vec3 origin = bbox_lower;
		glm::vec3 bbox_size = bbox_upper - bbox_lower;
		float dps = glm::max( glm::max( bbox_size.x, bbox_size.y ), bbox_size.z ) / (float)gridRes;

        for( int i = 0; i < vertices.size(); i += 3 )
        {
			float3 v0 = toFloat3( vertices[i] );
			float3 v1 = toFloat3( vertices[i + 1] );
			float3 v2 = toFloat3( vertices[i + 2] );

			float3 c0 = toFloat3( vcolors[i] );
			float3 c1 = toFloat3( vcolors[i + 1] );
			float3 c2 = toFloat3( vcolors[i + 2] );

            VTContext context( v0, v1, v2, sixSeparating, { origin.x, origin.y, origin.z }, dps, gridRes );
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
							if( showVertexColor )
							{
								float3 bc = closestBarycentricCoordinateOnTriangle( v0, v1, v2, p );
								float3 bColor = bc.x * c1 + bc.y * c2 + bc.z * c0;
								glm::u8vec4 voxelColor = { bColor.x * 255.0f + 0.5f, bColor.y * 255.0f + 0.5f, bColor.z * 255.0f + 0.5f, 255 };
								voxelColors.push_back( voxelColor );
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
		if( drawFace )
		{
			drawVoxelsFace( mortonVoxels, voxelColors, origin, dps );
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
		ImGui::Checkbox( "showVertexColor", &showVertexColor );
		
		ImGui::SeparatorText( "Save" );
		if( ImGui::Button( "Save As Mesh" ) )
		{
			VoxelMeshWriter writer;
			std::set<uint64_t> voxels( mortonVoxels.begin(), mortonVoxels.end() );

			saveVoxelsAsMesh( GetDataPath( "vox.ply" ).c_str(), std::vector<uint64_t>( voxels.begin(), voxels.end() ), origin, dps );
			//for (auto m : voxels)
			//{
			//	uint32_t x, y, z;
			//	decodeMortonCode_PEXT( m, &x, &y, &z );
			//	writer.add( { origin.x + x * dps, origin.y + y * dps, origin.z + z * dps }, dps );
			//}

			//writer.savePLY( GetDataPath( "vox.ply" ).c_str() );
		}

		ImGui::End();

		EndImGui();
	}

	pr::CleanUp();
}