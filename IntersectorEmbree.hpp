#pragma once

#include <stdint.h>
#include <glm/glm.hpp>
#include <set>

#include <embree4/rtcore.h>
#include <embree4/rtcore_ray.h>

#include "morton.hpp"

struct UserGeom
{
    glm::vec3 o;
    float radius;
};
inline void EmbreeErorrHandler(void* userPtr, RTCError code, const char* str)
{
    printf("Embree Error [%d] %s\n", code, str);
}
inline void boundsFunction( const RTCBoundsFunctionArguments* args )
{
    const UserGeom& geom = ((const UserGeom*)args->geometryUserPtr)[args->primID];
    RTCBounds* bounds_o = args->bounds_o;
    bounds_o->lower_x = geom.o.x - geom.radius;
    bounds_o->lower_y = geom.o.y - geom.radius;
    bounds_o->lower_z = geom.o.z - geom.radius;
    bounds_o->upper_x = geom.o.x + geom.radius;
    bounds_o->upper_y = geom.o.y + geom.radius;
    bounds_o->upper_z = geom.o.z + geom.radius;
}

inline void intersectFunc( const RTCIntersectFunctionNArguments* args )
{
	UserGeom* ptr = (UserGeom*)args->geometryUserPtr;
	RTCRayHit* ray = (RTCRayHit*)( args->rayhit );
	RTCHit* hit = (RTCHit*)&ray->hit;
	uint32_t primID = args->primID;
	uint32_t geomID = args->geomID;

	glm::vec3 ro = { ray->ray.org_x,
					 ray->ray.org_y,
					 ray->ray.org_z };
	glm::vec3 rd = { ray->ray.dir_x,
					 ray->ray.dir_y,
					 ray->ray.dir_z };

	glm::vec3 lower = ptr[primID].o - glm::vec3( ptr[primID].radius );
	glm::vec3 upper = ptr[primID].o + glm::vec3( ptr[primID].radius );
	glm::vec3 one_over_rd = glm::vec3( 1.0f ) / rd;

	glm::vec3 t0 = ( lower - ro ) * one_over_rd;
	glm::vec3 t1 = ( upper - ro ) * one_over_rd;
	glm::vec3 tlower = glm::min( t0, t1 );
	glm::vec3 tupper = glm::max( t0, t1 );
	float S_lmax = glm::max( glm::max( tlower.x, tlower.y ), tlower.z );
	float S_umin = glm::min( glm::min( tupper.x, tupper.y ), tupper.z );
	if( glm::min( S_umin, ray->ray.tfar ) < glm::max( S_lmax, 0.0f ) )
		return;

	ray->ray.tfar = S_lmax;
	ray->hit.primID = primID;
	ray->hit.geomID = geomID;

	if( S_lmax == tlower.x )
	{
		ray->hit.Ng_x = 1;
	}
	else if( S_lmax == tlower.y )
	{
		ray->hit.Ng_x = 2;
	}
	else
	{
		ray->hit.Ng_x = 0;
	}
}

inline bool memoryMonitor(
    void* userPtr,
    ssize_t bytes,
    bool post
)
{
    ssize_t* o = (ssize_t*)userPtr;
    *o += bytes;

    return true;
}

class IntersectorEmbree
{
public:
	IntersectorEmbree()
	{
		_embreeDevice = std::shared_ptr<RTCDeviceTy>( rtcNewDevice( "set_affinity=1" ), rtcReleaseDevice );
		rtcSetDeviceErrorFunction( _embreeDevice.get(), EmbreeErorrHandler, nullptr );
		rtcSetDeviceMemoryMonitorFunction( _embreeDevice.get(), memoryMonitor, &_bytes );
	}

    void build( const std::vector<uint64_t>& mortonVoxels, const glm::vec3& origin, float dps )
    {
		_embreeScene = std::shared_ptr<RTCSceneTy>( rtcNewScene( _embreeDevice.get() ), rtcReleaseScene );
		rtcSetSceneBuildQuality( _embreeScene.get(), RTC_BUILD_QUALITY_HIGH );

		_geometries.clear();
		for( uint64_t morton : mortonVoxels )
		{
			uint32_t x, y, z;
			decodeMortonCode_PEXT( morton, &x, &y, &z );

			UserGeom geom;
			geom.o.x = origin.x + dps * ( (float)x + 0.5f );
			geom.o.y = origin.y + dps * ( (float)y + 0.5f );
			geom.o.z = origin.z + dps * ( (float)z + 0.5f );
			geom.radius = dps * 0.5f;
			_geometries.push_back( geom );
		}

		RTCGeometry g = rtcNewGeometry( _embreeDevice.get(), RTC_GEOMETRY_TYPE_USER );
		rtcSetGeometryUserPrimitiveCount( g, _geometries.size() );
		rtcSetGeometryUserData( g, _geometries.data() );
		rtcSetGeometryBoundsFunction( g, boundsFunction, nullptr );
		rtcSetGeometryIntersectFunction( g, intersectFunc );

		rtcCommitGeometry( g );
		rtcAttachGeometry( _embreeScene.get(), g );
		rtcReleaseGeometry( g );

		rtcCommitScene( _embreeScene.get() );
    }
	void intersect( glm::vec3 ro, glm::vec3 rd, float* t, int* nMajor )
    {
		RTCRayHit rayHit = {};
		rayHit.ray.org_x = ro.x;
		rayHit.ray.org_y = ro.y;
		rayHit.ray.org_z = ro.z;
		rayHit.ray.dir_x = rd.x;
		rayHit.ray.dir_y = rd.y;
		rayHit.ray.dir_z = rd.z;
		rayHit.ray.tnear = 0.0f;
		rayHit.ray.tfar = *t;
		rayHit.ray.mask = 0xFFFFFFFF;
		rayHit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
		rayHit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;
		rayHit.hit.primID = RTC_INVALID_GEOMETRY_ID;

		rtcIntersect1( _embreeScene.get(), &rayHit );

		*t = rayHit.ray.tfar;
		*nMajor = (int)rayHit.hit.Ng_x;
    }
	
	uint64_t getMemoryConsumption()
	{
		return _bytes;
	}

	std::shared_ptr<RTCDeviceTy> _embreeDevice;
	std::shared_ptr<RTCSceneTy> _embreeScene;

	std::vector<UserGeom> _geometries;

	ssize_t _bytes = 0;
};