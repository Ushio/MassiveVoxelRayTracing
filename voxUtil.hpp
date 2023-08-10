#pragma once

#include <vector>
#include "pr.hpp"
#include "morton.hpp"

inline void trianglesFlattened( 
	std::shared_ptr<pr::FScene> scene, 
	std::vector<glm::vec3>* vertices, 
	std::vector<glm::vec3>* vcolors,
	std::vector<glm::vec3>* vemissions )
{
    using namespace pr;
    vertices->clear();
	vcolors->clear();
	vcolors->clear();
	vemissions->clear();

    scene->visitPolyMesh( [&]( std::shared_ptr<const FPolyMeshEntity> polymesh ) {
        ColumnView<int32_t> faceCounts( polymesh->faceCounts() );
	    ColumnView<int32_t> indices( polymesh->faceIndices() );
	    ColumnView<glm::vec3> positions( polymesh->positions() );

		const AttributeSpreadsheet* spreadsheet = polymesh->attributeSpreadsheet( AttributeSpreadsheetType::Points );
		const AttributeVector3Column* colorAttirb = spreadsheet->columnAsVector3( "Cd" );
		const AttributeVector3Column* emissionAttirb = spreadsheet->columnAsVector3( "Emission" );

		glm::mat4 m = polymesh->localToWorld();
	    for( int i = 0; i < faceCounts.count(); i++ )
	    {
		    PR_ASSERT( faceCounts[i] == 3 ); // no quad support now.
		    for( int j = 0; j < 3; ++j )
		    {
			    int index = indices[i * 3 + j];
				vertices->push_back( m * glm::vec4( positions[index], 1.0f ) );
				if( colorAttirb )
				{
					//vcolors->push_back( colorAttirb->get( i * 3 + j ) );
					vcolors->push_back( colorAttirb->get( index ) );
				}
				else
				{
					vcolors->push_back( glm::vec3( 1.0f ) );
				}
				if( emissionAttirb )
				{
					//vemissions->push_back( emissionAttirb->get( i * 3 + j ) );
					vemissions->push_back( emissionAttirb->get( index ) );
				}
				else
				{
					vemissions->push_back( glm::vec3( 0.0f ) );
				}
		    }
	    }
    } );
}
inline void getBoundingBox( const std::vector<glm::vec3>& vertices, glm::vec3* lower, glm::vec3* upper )
{
	glm::vec3 bbox_lower = glm::vec3( FLT_MAX );
	glm::vec3 bbox_upper = glm::vec3( -FLT_MAX );
	for( int i = 0; i < vertices.size(); i++ )
	{
		bbox_lower = glm::min( bbox_lower, vertices[i] );
		bbox_upper = glm::max( bbox_upper, vertices[i] );
	}
	*lower = bbox_lower;
	*upper = bbox_upper;
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
