#pragma once

#include "vectorMath.hpp"

DEVICE inline float2 ss_floor( float2 x )
{
	return { ss_floor( x.x ), ss_floor( x.y ) };
}

DEVICE inline float3 ss_floor( float3 x )
{
	return { ss_floor( x.x ), ss_floor( x.y ), ss_floor( x.z ) };
}

DEVICE inline float2 project2plane( float3 p, int axis )
{
    float2 r;
    switch (axis)
    {
    case 0: // z axis
        r.x = p.x;
        r.y = p.y;
        break;
    case 1: // x axis
        r.x = p.y;
        r.y = p.z;
        break;
    case 2: // y axis
        r.x = p.z;
        r.y = p.x;
        break;
    }
    return r;
}

DEVICE inline float project2plane_reminder( float3 p, int axis )
{
    switch (axis)
    {
    case 0:
        return p.z;
    case 1:
        return p.x;
    case 2:
        return p.y;
    }
    return 0.0f;
}

DEVICE inline int2 project2plane( int3 p, int axis )
{
    int2 r;
    switch (axis)
    {
    case 0: // z axis
        r.x = p.x;
        r.y = p.y;
        break;
    case 1: // x axis
        r.x = p.y;
        r.y = p.z;
        break;
    case 2: // y axis
        r.x = p.z;
        r.y = p.x;
        break;
    }
    return r;
}
DEVICE inline int project2plane_reminder( int3 p, int axis )
{
    switch (axis)
    {
    case 0:
        return p.z;
    case 1:
        return p.x;
    case 2:
        return p.y;
    }
    return 0;
}

DEVICE inline int majorAxis( float3 d )
{
    float x = ss_abs(d.x);
    float y = ss_abs(d.y);
    float z = ss_abs(d.z);
    if (x < y)
    {
        return y < z ? 0 : 2;
    }
    return x < z ? 0 : 1;
}

DEVICE inline float3 unProjectPlane( float2 p, float reminder, int axis )
{
    switch (axis)
    {
    case 0:
		return { p.x, p.y, reminder };
    case 1:
		return { reminder, p.x, p.y };
    case 2:
		return { p.y, reminder, p.x };
    }
	return { 0.0f, 0.0f, 0.0f };
}

DEVICE inline int3 unProjectPlane( int2 p, int reminder, int axis )
{
    switch (axis)
    {
    case 0:
		return { p.x, p.y, reminder };
    case 1:
		return { reminder, p.x, p.y };
    case 2:
		return { p.y, reminder, p.x };
    }
	return { 0, 0, 0 };
}

struct VTContext
{
    int major;

    int2 lower_xy;
    int2 upper_xy;
    int lower_z;
    int upper_z;

    float d_consts[3 /*axis*/][3 /*edge*/];
    float nesx[3][3];
    float nesy[3][3];

    float2 origin_xy;
    float origin_z;

    float kx;
    float ky;
    float constant_max;
    float constant_min;
    float constant_six;

    DEVICE VTContext( float3 v0, float3 v1, float3 v2, bool sixSeparating, float3 origin, float dps, int gridRes )
    {
        float3 e01 = v1 - v0;
        float3 e12 = v2 - v1;
        float3 n = cross(e01, e12);
        major = majorAxis(n);

        float3 bbox_lower = fminf( fminf( v0, v1 ), v2 );
		float3 bbox_upper = fmaxf( fmaxf( v0, v1 ), v2 );
		float3 lowerf = floorf( ( bbox_lower - origin ) / dps );
		float3 upperf = floorf( ( bbox_upper - origin ) / dps );
		int3 lower = { (int)lowerf.x, (int)lowerf.y, (int)lowerf.z };
		int3 upper = { (int)upperf.x, (int)upperf.y, (int)upperf.z };
		lower = maxi( lower, int3{ 0, 0, 0 } );
		upper = mini( upper, int3{ gridRes - 1, gridRes - 1, gridRes - 1 } );

        lower_xy = project2plane(lower, major);
        upper_xy = project2plane(upper, major);
        lower_z = project2plane_reminder(lower, major);
        upper_z = project2plane_reminder(upper, major);

        for (int axis = 0; axis < 3; axis++)
        {
			float2 dp_proj = float2{ dps, dps };
            float2 vs_proj[3] = {
                project2plane(v0, axis),
                project2plane(v1, axis),
                project2plane(v2, axis),
            };
            float reminder = project2plane_reminder(n, axis);
            float n_sign = 0.0f < reminder ? 1.0f : -1.0f;

            for (int edge = 0; edge < 3; edge++)
            {
                float2 a = vs_proj[edge];
                float2 b = vs_proj[(edge + 1) % 3];
                float2 e = b - a;
				float2 ne = float2{ -e.y, e.x } * n_sign;
                nesx[axis][edge] = ne.x;
                nesy[axis][edge] = ne.y;

                float d_const;
                if (sixSeparating == false)
                {
					d_const = ss_max( ne.x * dp_proj.x, 0.0f )
                        + ss_max(ne.y * dp_proj.y, 0.0f)
                        - dot(ne, a);
                }
                else
                {
                    d_const = dot(ne, dp_proj * 0.5f - a)
                        + 0.5f * dps * ss_max(ss_abs(ne.x), ss_abs(ne.y));
                }
                d_consts[axis][edge] = d_const;
            }
        }

        origin_xy = project2plane(origin, major);
        origin_z = project2plane_reminder(origin, major);

        float2 v0_xy = project2plane(v0, major);
        float v0_z = project2plane_reminder(v0, major);

        float2 n_xy = project2plane(n, major);
        float n_z = project2plane_reminder(n, major);

        kx = -n_xy.x / n_z;
        ky = -n_xy.y / n_z;
        float K = -kx * v0_xy.x - ky * v0_xy.y + v0_z;
        constant_max =
            K
            + dps * (ss_max(kx, 0.0f) + ss_max(ky, 0.0f));
        constant_min =
            K
            + dps * (ss_min(kx, 0.0f) +ss_min(ky, 0.0f));
        constant_six =
            K
            + 0.5f * dps * (kx + ky);
    }
	DEVICE int2 xRangeInclusive() const
    {
        return { lower_xy.x, upper_xy.x };
    }

    DEVICE int2 yRangeInclusive( int x, float dps ) const
    {
        float xcoord = origin_xy.x + x * dps;

        float miny = -3.402823466e+38F;
		float maxy = 3.402823466e+38F;
        for (int edge = 0; edge < 3; edge++)
        {
            float nex = nesx[major][edge];
            float ney = nesy[major][edge];
            float d_const = d_consts[major][edge];
            if( ney == 0.0f )
            {
                if( -nex * xcoord <= d_const )
                {
                    continue;
                }
                else
                {
					return { 1, -1 };
                }
            }
            float k = -( xcoord * nex + d_const ) / ney;
            if( 0.0f < ney )
            {
				miny = ss_max( miny, k );
            }
            else
            {
				maxy = ss_min( maxy, k );
            }
        }
        float minIndexF = ss_max( (miny - origin_xy.y) / dps, -2147483648.0f /* valid int cast min */);
        float maxIndexF = ss_min( (maxy - origin_xy.y) / dps, 2147483520.0f /* valid int cast max */ );
        int lowerY = (int)ss_ceil( minIndexF );
        int upperY = (int)ss_floor( maxIndexF );
        lowerY = ss_max(lowerY, lower_xy.y );
        upperY = ss_min(upperY, upper_xy.y );
		return { lowerY, upperY };
    }
	DEVICE int2 zRangeInclusive( int x, int y, float dps, bool sixSeparating ) const
    {
		float2 o_xy = origin_xy + float2{ dps * x, dps * y };
        float var = kx * o_xy.x + ky * o_xy.y;

        int zmin;
        int zmax;

        if( sixSeparating )
        {
            float tsix = var + constant_six;
            float indexf = (tsix - origin_z) / dps;
            float zf = ss_floor(indexf);
            int z = (int)zf;
            zmin = indexf == zf ? z - 1 : z;
            zmax = z;
        }
        else
        {
            float tmax = var + constant_max;
            float tmin = var + constant_min;
            zmin = (int)(ss_floor((tmin - origin_z) / dps));
            zmax = (int)(ss_floor((tmax - origin_z) / dps));
        }

        zmin = ss_max(zmin, lower_z );
        zmax = ss_min(zmax, upper_z );

        return { zmin, zmax };
    }
	DEVICE float3 p( int x, int y, int z, float dps ) const
    {
		float2 p_proj = origin_xy + float2{ dps * x, dps * y };
        float reminder = origin_z + (float)z * dps;
        return unProjectPlane(p_proj, reminder, major );
    }
	DEVICE int3 i( int x, int y, int z ) const
    {
		return unProjectPlane( { x, y }, z, major );
    }

    DEVICE bool intersect( float3 p ) const
    {
        if( lower_z == upper_z )
        {
            return true;
        }

        for (int axis = 0; axis < 3; axis++)
        {
            if (axis == major)
                continue;

            float2 p_proj = project2plane( p, axis );
            for (int edge = 0; edge < 3; edge++)
            {
                float nex = nesx[axis][edge];
                float ney = nesy[axis][edge];
                float d = nex * p_proj.x + ney * p_proj.y + d_consts[axis][edge];
                if (d < 0.0f)
                {
                    return false;
                }
            }
        }
        return true;
    }
};