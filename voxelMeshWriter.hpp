#pragma once
#include <stdlib.h>
#include <vector>

class VoxelMeshWriter
{
public:
	void add( glm::vec3 p, float dps )
	{
		points.push_back( p );
		points.push_back( p + glm::vec3( dps, 0, 0 ) );
		points.push_back( p + glm::vec3( dps, 0, dps ) );
		points.push_back( p + glm::vec3( 0, 0, dps ) );
		points.push_back( p + glm::vec3( 0, dps, 0 ) );
		points.push_back( p + glm::vec3( dps, dps, 0 ) );
		points.push_back( p + glm::vec3( dps, dps, dps ) );
		points.push_back( p + glm::vec3( 0, dps, dps ) );
	}

	void savePLY( const char* file )
	{
		FILE* fp = fopen( file, "wb" );

		int nVoxels = points.size() / 8;

		// PLY header
		fprintf( fp, "ply\n" );
		fprintf( fp, "format binary_little_endian 1.0\n" );
		fprintf( fp, "element vertex %llu\n", points.size() );
		fprintf( fp, "property float x\n" );
		fprintf( fp, "property float y\n" );
		fprintf( fp, "property float z\n" );
		fprintf( fp, "element face %d\n", nVoxels * 6 );
		fprintf( fp, "property list uchar uint vertex_indices\n" );
		fprintf( fp, "end_header\n" );

		// Write vertices
		fwrite( points.data(), sizeof( glm::vec3 ) * points.size(), 1, fp );

		std::vector<uint8_t> bytes( ( 1 + sizeof( uint32_t ) * 4 ) * 6 * nVoxels );
		uint32_t head = 0;
		for( int i = 0; i < nVoxels; i++ )
		{
			uint32_t i0 = i * 8;
			uint32_t i1 = i * 8 + 1;
			uint32_t i2 = i * 8 + 2;
			uint32_t i3 = i * 8 + 3;
			uint32_t i4 = i * 8 + 4;
			uint32_t i5 = i * 8 + 5;
			uint32_t i6 = i * 8 + 6;
			uint32_t i7 = i * 8 + 7;

#define F( a, b, c, d )                          \
	bytes[head++] = 4;\
	memcpy( &bytes[head], &a, sizeof( uint32_t ) ); head += 4;\
	memcpy( &bytes[head], &b, sizeof( uint32_t ) ); head += 4;\
	memcpy( &bytes[head], &c, sizeof( uint32_t ) ); head += 4;\
	memcpy( &bytes[head], &d, sizeof( uint32_t ) ); head += 4;

			// Left Hand
			F( i3, i2, i1, i0 );
			F( i4, i5, i6, i7 );
			F( i0, i1, i5, i4 );
			F( i1, i2, i6, i5 );
			F( i2, i3, i7, i6 );
			F( i3, i0, i4, i7 );
#undef F
		}
		fwrite( bytes.data(), bytes.size(), 1, fp );
		fclose( fp );
	}
	std::vector<glm::vec3> points;
};