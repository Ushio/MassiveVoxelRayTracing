#pragma once

#include <vector>
#include "pr.hpp"

inline void trianglesFlattened( std::shared_ptr<pr::FScene> scene, std::vector<glm::vec3>* vertices, std::vector<glm::vec3>* vcolors )
{
    using namespace pr;
    vertices->clear();
	vcolors->clear();

    scene->visitPolyMesh( [&]( std::shared_ptr<const FPolyMeshEntity> polymesh ) {
        ColumnView<int32_t> faceCounts( polymesh->faceCounts() );
	    ColumnView<int32_t> indices( polymesh->faceIndices() );
	    ColumnView<glm::vec3> positions( polymesh->positions() );

		const AttributeSpreadsheet* spreadsheet = polymesh->attributeSpreadsheet( AttributeSpreadsheetType::Vertices );
		const AttributeVector4Column* colorAttirb = spreadsheet->columnAsVector4( "Color" );

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
					vcolors->push_back( colorAttirb->get( i * 3 + j ) );
				}
				else
				{
					vcolors->push_back( glm::vec3( 1.0f ) );
				}
		    }
	    }
    } );
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
// return { U, V, W }, where U * v1 + V * v2 + W * v0 is the point
inline glm::vec3 closestBarycentricCoordinateOnTriangle( glm::vec3 v0, glm::vec3 v1, glm::vec3 v2, glm::vec3 P )
{
	glm::vec3 d0 = v0 - P;
	glm::vec3 d1 = v1 - P;
	glm::vec3 d2 = v2 - P;
	glm::vec3 e0 = v2 - v0;
	glm::vec3 e1 = v0 - v1;
	glm::vec3 e2 = v1 - v2;
	glm::vec3 Ng = glm::cross( e2, e0 );

	// bc inside the triangle
	// barycentric coordinate from tetrahedron volumes
	float U = glm::dot( cross( d2, d0 ), Ng );
	float V = glm::dot( cross( d0, d1 ), Ng );
	float W = glm::dot( cross( d1, d2 ), Ng );

	// bc outside the triangle
	if( U < 0.0f )
	{
		V = glm::dot( -d0, e0 );
		W = glm::dot( d2, e0 );
	}
	else if( V < 0.0f )
	{
		W = glm::dot( -d1, e1 );
		U = glm::dot( d0, e1 );
	}
	else if( W < 0.0f )
	{
		U = glm::dot( -d2, e2 );
		V = glm::dot( d1, e2 );
	}

	glm::vec3 bc = glm::max( glm::vec3( 0.0f ), { U, V, W } );
	return bc / ( bc.x + bc.y + bc.z );
}