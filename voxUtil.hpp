#pragma once

#include <vector>
#include "pr.hpp"

inline void trianglesFlattened( std::shared_ptr<pr::FScene> scene, std::vector<glm::vec3>* vertices )
{
    using namespace pr;
    vertices->clear();

    scene->visitPolyMesh( [&]( std::shared_ptr<const FPolyMeshEntity> polymesh ) {
        ColumnView<int32_t> faceCounts( polymesh->faceCounts() );
	    ColumnView<int32_t> indices( polymesh->faceIndices() );
	    ColumnView<glm::vec3> positions( polymesh->positions() );

		glm::mat4 m = polymesh->localToWorld();
	    for( int i = 0; i < faceCounts.count(); i++ )
	    {
		    PR_ASSERT( faceCounts[i] == 3 ); // no quad support now.
		    for( int j = 0; j < 3; ++j )
		    {
			    int index = indices[i * 3 + j];
				vertices->push_back( m * glm::vec4( positions[index], 1.0f ) );
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
