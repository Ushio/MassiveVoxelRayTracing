inline float ss_floor( float value )
{
    float d;
    _mm_store_ss(&d, _mm_floor_ss(_mm_setzero_ps(), _mm_set_ss(value)));
    return d;
}
inline float ss_ceil(float value)
{
    float d;
    _mm_store_ss(&d, _mm_ceil_ss(_mm_setzero_ps(), _mm_set_ss(value)));
    return d;
}
inline glm::vec2 ss_floor(glm::vec2 x)
{
    return glm::vec2(ss_floor(x.x), ss_floor(x.y));
}

inline glm::vec3 ss_floor(glm::vec3 x)
{
    return glm::vec3(ss_floor(x.x), ss_floor(x.y), ss_floor(x.z));
}

inline glm::vec2 project2plane(glm::vec3 p, int axis)
{
    glm::vec2 r;
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

inline float project2plane_reminder(glm::vec3 p, int axis)
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

inline glm::ivec2 project2plane( glm::ivec3 p, int axis )
{
    glm::ivec2 r;
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
inline int project2plane_reminder( glm::ivec3 p, int axis)
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

inline int majorAxis(glm::vec3 d)
{
    float x = glm::abs(d.x);
    float y = glm::abs(d.y);
    float z = glm::abs(d.z);
    if (x < y)
    {
        return y < z ? 0 : 2;
    }
    return x < z ? 0 : 1;
}

inline glm::vec3 unProjectPlane(glm::vec2 p, float reminder, int axis)
{
    switch (axis)
    {
    case 0:
        return glm::vec3(p.x, p.y, reminder);
    case 1:
        return glm::vec3(reminder, p.x, p.y );
    case 2:
        return glm::vec3(p.y, reminder, p.x );
    }
    return glm::vec3(0.0f, 0.0f, 0.0f);
}

inline glm::ivec3 unProjectPlane(glm::ivec2 p, int reminder, int axis)
{
    switch (axis)
    {
    case 0:
        return glm::ivec3(p.x, p.y, reminder);
    case 1:
        return glm::ivec3(reminder, p.x, p.y );
    case 2:
        return glm::ivec3(p.y, reminder, p.x );
    }
    return glm::ivec3(0, 0, 0);
}

struct VTContext
{
    int major;

    glm::ivec2 lower_xy;
    glm::ivec2 upper_xy;
    int lower_z;
    int upper_z;

    float d_consts[3 /*axis*/][3 /*edge*/];
    float nesx[3][3];
    float nesy[3][3];

    glm::vec2 origin_xy;
    float origin_z;

    float kx;
    float ky;
    float constant_max;
    float constant_min;
    float constant_six;

    VTContext( glm::vec3 v0, glm::vec3 v1, glm::vec3 v2, bool sixSeparating, glm::vec3 origin, float dps, int gridRes )
    {
        glm::vec3 e01 = v1 - v0;
        glm::vec3 e12 = v2 - v1;
        glm::vec3 n = glm::cross(e01, e12);
        major = majorAxis(n);

        glm::vec3 bbox_lower = glm::min(glm::min(v0, v1), v2);
        glm::vec3 bbox_upper = glm::max(glm::max(v0, v1), v2);
        glm::ivec3 lower = glm::ivec3(ss_floor((bbox_lower - origin) / dps));
        glm::ivec3 upper = glm::ivec3(ss_floor((bbox_upper - origin) / dps));
        lower = glm::max(lower, glm::ivec3(0, 0, 0));
        upper = glm::min(upper, glm::ivec3(gridRes - 1, gridRes - 1, gridRes - 1));

        lower_xy = project2plane(lower, major);
        upper_xy = project2plane(upper, major);
        lower_z = project2plane_reminder(lower, major);
        upper_z = project2plane_reminder(upper, major);

        for (int axis = 0; axis < 3; axis++)
        {
            glm::vec2 dp_proj = glm::vec2(dps, dps);
            glm::vec2 vs_proj[3] = {
                project2plane(v0, axis),
                project2plane(v1, axis),
                project2plane(v2, axis),
            };
            float reminder = project2plane_reminder(n, axis);
            float n_sign = 0.0f < reminder ? 1.0f : -1.0f;

            for (int edge = 0; edge < 3; edge++)
            {
                glm::vec2 a = vs_proj[edge];
                glm::vec2 b = vs_proj[(edge + 1) % 3];
                glm::vec2 e = b - a;
                glm::vec2 ne = glm::vec2(-e.y, e.x) * n_sign;
                nesx[axis][edge] = ne.x;
                nesy[axis][edge] = ne.y;

                float d_const;
                if (sixSeparating == false)
                {
                    d_const = glm::max(ne.x * dp_proj.x, 0.0f)
                        + glm::max(ne.y * dp_proj.y, 0.0f)
                        - glm::dot(ne, a);
                }
                else
                {
                    d_const = glm::dot(ne, dp_proj * 0.5f - a)
                        + 0.5f * dps * glm::max(glm::abs(ne.x), glm::abs(ne.y));
                }
                d_consts[axis][edge] = d_const;
            }
        }

        origin_xy = project2plane(origin, major);
        origin_z = project2plane_reminder(origin, major);

        glm::vec2 v0_xy = project2plane(v0, major);
        float v0_z = project2plane_reminder(v0, major);

        glm::vec2 n_xy = project2plane(n, major);
        float n_z = project2plane_reminder(n, major);

        kx = -n_xy.x / n_z;
        ky = -n_xy.y / n_z;
        float K = -kx * v0_xy.x - ky * v0_xy.y + v0_z;
        constant_max =
            K
            + dps * (glm::max(kx, 0.0f) + glm::max(ky, 0.0f));
        constant_min =
            K
            + dps * (glm::min(kx, 0.0f) + glm::min(ky, 0.0f));
        constant_six =
            K
            + 0.5f * dps * (kx + ky);
    }
    glm::ivec2 xRangeInclusive() const
    {
        return { lower_xy.x, upper_xy.x };
    }

    glm::ivec2 yRangeInclusive( int x, float dps ) const
    {
        float xcoord = origin_xy.x + x * dps;

        float miny = -FLT_MAX; // valid int cast min
        float maxy = FLT_MAX;  // valid int cast max
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
                    return glm::ivec2(1, -1);
                }
            }
            float k = -( xcoord * nex + d_const ) / ney;
            if( 0.0f < ney )
            {
                miny = glm::max( miny, k );
            }
            else
            {
                maxy = glm::min( maxy, k );
            }
        }
        float minIndexF = glm::max( (miny - origin_xy.y) / dps, -2147483648.0f /* valid int cast min */);
        float maxIndexF = glm::min( (maxy - origin_xy.y) / dps, 2147483520.0f /* valid int cast max */ );
        int lowerY = (int)ss_ceil( minIndexF );
        int upperY = (int)ss_floor( maxIndexF );
        lowerY = glm::max(lowerY, lower_xy.y );
        upperY = glm::min(upperY, upper_xy.y );
        return glm::ivec2( lowerY, upperY );
    }
    glm::ivec2 zRangeInclusive(int x, int y, float dps, bool sixSeparating)const
    {
        glm::vec2 o_xy = origin_xy + glm::vec2(dps * x, dps * y);
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

        zmin = glm::max(zmin, lower_z );
        zmax = glm::min(zmax, upper_z );

        return glm::ivec2(zmin, zmax);
    }
    glm::vec3 p( int x, int y, int z, float dps ) const
    {
        glm::vec2 p_proj = origin_xy + glm::vec2( dps * x, dps * y );
        float reminder = origin_z + (float)z * dps;
        return unProjectPlane(p_proj, reminder, major );
    }
    glm::ivec3 i( int x, int y, int z ) const
    {
        return unProjectPlane(glm::ivec2(x, y), z, major);
    }

    bool intersect( glm::vec3 p ) const
    {
        if( lower_z == upper_z )
        {
            return true;
        }

        for (int axis = 0; axis < 3; axis++)
        {
            if (axis == major)
                continue;

            glm::vec2 p_proj = project2plane( p, axis );
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