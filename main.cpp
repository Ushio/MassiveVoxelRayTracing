#include "pr.hpp"
#include <iostream>
#include <memory>

glm::vec2 project2plane(glm::vec3 p, int axis)
{
    glm::vec2 r;
    switch (axis)
    {
    case 0: // z axis
        r.x = p.x;
        r.y = p.y;
        break;
    case 1: // y axis
        r.x = p.y;
        r.y = p.z;
        break;
    case 2: // x axis
        r.x = p.z;
        r.y = p.x;
        break;
    }
    return r;
}
float project2plane_reminder(glm::vec3 p, int axis)
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

bool overlapAABB(glm::vec3 lowerA, glm::vec3 upperA, glm::vec3 lowerB, glm::vec3 upperB)
{
    if (upperA.x < lowerB.x || upperA.y < lowerB.y || upperA.z < lowerB.z)
    {
        return false;
    }

    if (upperB.x < lowerA.x || upperB.y < lowerA.y || upperB.z < lowerA.z)
    {
        return false;
    }
    return true;
}

glm::ivec2 project2plane( glm::ivec3 p, int axis )
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
int majorAxis(glm::vec3 d)
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

glm::vec3 unProjectPlane(glm::vec2 p, float reminder, int axis)
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

glm::ivec3 unProjectPlane(glm::ivec2 p, int reminder, int axis)
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

//static bool planeVoxelTest = true;

struct VoxelTriangleIntersector
{
    glm::vec3 n;
    glm::vec3 triangle_lower;
    glm::vec3 triangle_upper;

    // Plane-Voxel test. This is handled by zRangeInclusive
    // float d1;
    // float d2;

    float d_consts[3 /*axis*/][3 /*edge*/];
    glm::vec2 nes[3 /*axis*/][3 /*edge*/];

    VoxelTriangleIntersector( glm::vec3 v0, glm::vec3 v1, glm::vec3 v2, bool sixSeparating, float dps )
    {
        glm::vec3 e01 = v1 - v0;
        glm::vec3 e12 = v2 - v1;
        n = glm::cross(e01, e12);

        glm::vec3 dp = glm::vec3(dps, dps, dps);

        triangle_lower = glm::min(glm::min(v0, v1), v2);
        triangle_upper = glm::max(glm::max(v0, v1), v2);

        glm::vec3 c = glm::vec3(
            0.0f < n.x ? dps : 0.0f,
            0.0f < n.y ? dps : 0.0f,
            0.0f < n.z ? dps : 0.0f
        );

        // Plane-Voxel test. This is handled by zRangeInclusive
        //if (sixSeparating == false)
        //{
        //    d1 = glm::dot(n, c - v0);
        //    d2 = glm::dot(n, dp - c - v0);
        //}
        //else
        //{
        //    float k1 = glm::dot(n, dp * 0.5f - v0);
        //    float k2 = 0.5f * dps * glm::max(glm::max(glm::abs(n.x), glm::abs(n.y)), glm::abs(n.z));
        //    d1 = k1 - k2;
        //    d2 = k1 + k2;
        //}

        for (int axis = 0; axis < 3; axis++)
        {
            glm::vec2 dp_proj = glm::vec2(dps, dps);
            glm::vec2 vs_proj[3] = {
                project2plane(v0, axis),
                project2plane(v1, axis),
                project2plane(v2, axis),
            };
            float reminder = project2plane_reminder(n, axis);

            for (int edge = 0; edge < 3; edge++)
            {
                glm::vec2 a = vs_proj[edge];
                glm::vec2 b = vs_proj[(edge + 1) % 3];
                glm::vec2 e = b - a;
                glm::vec2 ne = glm::vec2(-e.y, e.x);
                if (reminder < 0.0f) {
                    ne = -ne;
                }
                nes[axis][edge] = ne;

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
    }

    bool intersect( glm::vec3 p, float dps ) const
    {
        // Plane-Voxel test. This is handled by zRangeInclusive
        // float PoN = glm::dot(p, n);
        // if (0.0f < (PoN + d1) * (PoN + d2))
        // {
        //     return false;
        // }

        // bbox test for a corner case
        if (overlapAABB(p, p + glm::vec3( dps, dps, dps ), triangle_lower, triangle_upper) == false)
        {
            return false;
        }

        // projection test
        for (int axis = 0; axis < 3 ; axis++)
        {
            glm::vec2 p_proj = project2plane(p, axis);
            for (int edge = 0; edge < 3; edge++)
            {
                float d = glm::dot(nes[axis][edge], p_proj) + d_consts[axis][edge];
                if( d < 0.0f )
                {
                    return false;
                }
            }
        }
        return true;
    }
};

float ss_floor( float value )
{
    float d;
    _mm_store_ss(&d, _mm_floor_ss(_mm_setzero_ps(), _mm_set_ss(value)));
    return d;
}
float ss_ceil(float value)
{
    float d;
    _mm_store_ss(&d, _mm_ceil_ss(_mm_setzero_ps(), _mm_set_ss(value)));
    return d;
}
glm::vec2 ss_floor(glm::vec2 x)
{
    return glm::vec2(ss_floor(x.x), ss_floor(x.y));
}

glm::vec3 ss_floor(glm::vec3 x)
{
    return glm::vec3(ss_floor(x.x), ss_floor(x.y), ss_floor(x.z));
}

struct VoxelTriangleVisitor
{
    int major;
    float kx;
    float ky;
    float constant_max;
    float constant_min;
    float constant_six;

    glm::vec2 origin_xy;
    float origin_z;
    glm::ivec2 lower_xy;
    glm::ivec2 upper_xy;

    VoxelTriangleVisitor( glm::vec3 origin, glm::vec3 triangle_lower, glm::vec3 triangle_upper, glm::vec3 v0, glm::vec3 n, float dps, int gridRes )
    {
        major = majorAxis(n);
        origin_xy = project2plane(origin, major);
        origin_z = project2plane_reminder(origin, major);

        lower_xy = glm::ivec2(ss_floor((project2plane(triangle_lower, major) - origin_xy) / dps));
        upper_xy = glm::ivec2(ss_floor((project2plane(triangle_upper, major) - origin_xy) / dps));
        lower_xy = glm::max(lower_xy, glm::ivec2(0, 0));
        upper_xy = glm::min(upper_xy, glm::ivec2(gridRes - 1, gridRes - 1));

        glm::vec2 v0_xy = project2plane(v0, major);
        float v0_z = project2plane_reminder(v0, major);

        glm::vec2 n_xy = project2plane(n, major);
        float n_z = project2plane_reminder(n, major);

        kx = -n_xy.x / n_z;
        ky = -n_xy.y / n_z;
        float K = - kx * v0_xy.x - ky * v0_xy.y + v0_z;
        constant_max =
            K
            + dps * (glm::max(kx, 0.0f) + glm::max(ky, 0.0f));
        constant_min =
            K
            + dps * (glm::min(kx, 0.0f) + glm::min(ky, 0.0f));
        constant_six = 
            K
            + 0.5f * dps * ( kx + ky );
    }

    glm::ivec2 yRangeInclusive( const VoxelTriangleIntersector& intersector, int x, float dps )const
    {
        float xcoord = origin_xy.x + x * dps;
        float miny = -FLT_MAX;
        float maxy = FLT_MAX;
        for (int edge = 0; edge < 3; edge++)
        {
            glm::vec2 ne = intersector.nes[major][edge];
            if( glm::abs(ne.y) < FLT_EPSILON )
            {
                continue;
            }

            float k = -(xcoord * ne.x + intersector.d_consts[major][edge]) / ne.y;
            if (0.0f <= ne.y)
            {
                miny = glm::max( miny, k );
            }
            else
            {
                maxy = glm::min( maxy, k );
            }
        }
        int lowerY = glm::max((int)ss_ceil((miny - origin_xy.y) / dps), lower_xy.y);
        int upperY = glm::min((int)ss_floor((maxy - origin_xy.y) / dps), upper_xy.y);
        return glm::ivec2( lowerY, upperY );
    }

    // It returns exact z range for plane.
    glm::ivec2 zRangeInclusive( int x, int y, float dps, int gridRes, bool sixSeparating )const
    {
        glm::vec2 o_xy = p_projMajor(x, y, dps);
        float var = kx * o_xy.x + ky * o_xy.y;

        int lowerz;
        int upperz;

        if( sixSeparating )
        {
            float tsix = var + constant_six;
            int z = (int)ss_floor( (tsix - origin_z) / dps );
            lowerz = z;
            upperz = z;
        }
        else
        {
            float tmax = var + constant_max;
            float tmin = var + constant_min;
            lowerz = (int)(ss_floor((tmin - origin_z) / dps));
            upperz = (int)(ss_floor((tmax - origin_z) / dps));
        }

        // to limit the range inside the grid
        lowerz = glm::max( lowerz, 0 );
        upperz = glm::min( upperz, gridRes - 1);
        
        return glm::ivec2( lowerz, upperz );
    }
    glm::vec3 p( int x, int y, int reminder, float dps ) const
    {
        glm::vec2 o_xy = p_projMajor(x, y, dps);
        return unProjectPlane(o_xy, origin_z + (float)reminder * dps, major);
    }
    glm::vec2 p_projMajor(int x, int y, float dps) const
    {
        return origin_xy + glm::vec2(dps * x, dps * y);
    }
};

class SequencialHasher
{
public:
    void add(uint32_t xs, uint32_t resolution)
    {
        m_h += m_base * xs;
        m_base *= resolution;
    }
    uint32_t value() const { return m_h; }
private:
    uint32_t m_base = 1;
    uint32_t m_h = 0;
};

int main() {
    using namespace pr;

    Config config;
    config.ScreenWidth = 1920;
    config.ScreenHeight = 1080;
    config.SwapInterval = 1;
    Initialize(config);

    Camera3D camera;
    camera.origin = { 4, 4, 4 };
    camera.lookat = { 0, 0, 0 };
    camera.zUp = false;

    double e = GetElapsedTime();

    const char* input = "bunny.obj";
    // const char* input = "Tri.obj";
    SetDataDir(ExecutableDir());
    std::string errorMsg;
    std::shared_ptr<FScene> scene = ReadWavefrontObj( GetDataPath(input), errorMsg );

    while (pr::NextFrame() == false) {
        if (IsImGuiUsingMouse() == false) {
            UpdateCameraBlenderLike(&camera  );
        }

        ClearBackground(0.1f, 0.1f, 0.1f, 1);

        BeginCamera(camera);

        PushGraphicState();

        DrawGrid(GridAxis::XZ, 1.0f, 10, { 128, 128, 128 });
        DrawXYZAxis(1.0f);

        static double voxel_time = 0.0f;

#if 1
        static bool sixSeparating = true;
        static float dps = 0.1f;
        static glm::vec3 origin = { -2.0f, -2.0f, -2.0f };
        static int gridRes = 256;

        scene->visitPolyMesh([](std::shared_ptr<const FPolyMeshEntity> polymesh) {
            ColumnView<int32_t> faceCounts(polymesh->faceCounts());
            ColumnView<int32_t> indices(polymesh->faceIndices());
            ColumnView<glm::vec3> positions(polymesh->positions());

            // Geometry
            pr::PrimBegin(pr::PrimitiveMode::Lines);
            for (int i = 0; i < positions.count(); i++)
            {
                glm::vec3 p = positions[i];
                glm::ivec3 color = {255,255,255};
                pr::PrimVertex(p, { color });
            }
            int indexBase = 0;
            for (int i = 0; i < faceCounts.count(); i++)
            {
                int nVerts = faceCounts[i];
                for (int j = 0; j < nVerts; ++j)
                {
                    int i0 = indices[indexBase + j];
                    int i1 = indices[indexBase + (j + 1) % nVerts];
                    pr::PrimIndex(i0);
                    pr::PrimIndex(i1);
                }
                indexBase += nVerts;
            }
            pr::PrimEnd();

            // Assume Triangle

            glm::vec3 lower = glm::vec3( FLT_MAX, FLT_MAX, FLT_MAX );
            glm::vec3 upper = glm::vec3( -FLT_MAX, -FLT_MAX, -FLT_MAX );
            for (int i = 0; i < faceCounts.count(); i++)
            {
                for (int j = 0; j < 3; ++j)
                {
                    int index = indices[i * 3 + j];
                    lower = glm::min(lower, positions[index]);
                    upper = glm::max(upper, positions[index]);
                }
            }

            // for flat polygons on the edge
            {
                glm::vec3 s = ( upper - lower );
                lower -= s / 1024.0f;
                upper += s / 1024.0f;
            }

            // bounding box
            glm::vec3 size = upper - lower;

            float dps = glm::max(glm::max(size.x, size.y), size.z) / (float)gridRes;

            DrawAABB(lower, lower + glm::vec3(dps, dps, dps) * (float)gridRes, {255 ,0 ,0});

            // Voxelization
            static std::vector<char> voxels;
            voxels.resize(gridRes * gridRes * gridRes);
            std::fill( voxels.begin(), voxels.end(), 0 );

            Stopwatch voxelsw;

            glm::vec3 origin = lower;
            for (int i = 0; i < faceCounts.count(); i++)
            {
                glm::vec3 v0 = positions[indices[i * 3]];
                glm::vec3 v1 = positions[indices[i * 3 + 1]];
                glm::vec3 v2 = positions[indices[i * 3 + 2]];

                VoxelTriangleIntersector intersector(v0, v1, v2, sixSeparating, dps);

#if 0
                glm::ivec3 lower = glm::ivec3(ss_floor((intersector.triangle_lower - origin) / dps));
                glm::ivec3 upper = glm::ivec3(ss_floor((intersector.triangle_upper - origin) / dps));
                lower = glm::max(lower, glm::ivec3(0, 0, 0));
                upper = glm::min(upper, glm::ivec3(gridRes - 1, gridRes - 1, gridRes - 1));
                for (int x = lower.x; x <= upper.x; x++)
                for (int y = lower.y; y <= upper.y; y++)
                for (int z = lower.z; z <= upper.z; z++)
                {
                    glm::vec3 p = origin + glm::vec3(x, y, z) * dps;
                    bool overlap = intersector.intersect(p, dps);
                    if (overlap)
                    {
                        // DrawAABB(p, p + glm::vec3(dps, dps, dps), { 200 ,200 ,200 });
                        SequencialHasher h;
                        h.add(x, gridRes);
                        h.add(y, gridRes);
                        h.add(z, gridRes);
                        voxels[h.value()] = 1;
                    }
                }
#else
                VoxelTriangleVisitor visitor(origin, intersector.triangle_lower, intersector.triangle_upper, v0, intersector.n, dps, gridRes );
                for (int x = visitor.lower_xy.x; x <= visitor.upper_xy.x; x++)
                {
                    glm::ivec2 yrange = visitor.yRangeInclusive(intersector, x, dps );
                    for (int y = yrange.x; y <= yrange.y; y++)
                    {
                        glm::ivec2 zrange = visitor.zRangeInclusive(x, y, dps, gridRes, sixSeparating );
                        for (int z = zrange.x; z <= zrange.y; z++)
                        {
                            glm::vec3 p = visitor.p(x, y, z, dps);
                            bool overlap = intersector.intersect(p, dps);
                            if (overlap)
                            {
                                // DrawAABB(p, p + glm::vec3(dps, dps, dps), { 200 ,200 ,200 });
                                glm::ivec3 c = unProjectPlane(glm::ivec2(x, y), z, visitor.major);
                                SequencialHasher h;
                                h.add(c.x, gridRes);
                                h.add(c.y, gridRes);
                                h.add(c.z, gridRes);
                                voxels[h.value()] = 1;
                            }
                        }
                    }
                }
#endif
            } // face

            voxel_time = voxelsw.elapsed();

            // Draw
            for (int x = 0; x < gridRes; x++)
            for (int y = 0; y < gridRes; y++)
            for (int z = 0; z < gridRes; z++)
            {
                SequencialHasher h;
                h.add(x, gridRes);
                h.add(y, gridRes);
                h.add(z, gridRes);
                if (voxels[h.value()])
                {
                    glm::vec3 p = origin + glm::vec3(x, y, z) * dps;
                    DrawAABB(p, p + glm::vec3(dps, dps, dps), { 200 ,200 ,200 });
                }
            }
        });
#endif

#if 0
        float unit = 1.0f;
        static glm::vec3 v0 = { -unit, -unit - 0.3f, 0.0f };
        static glm::vec3 v1 = { unit , -unit, 0.0f };
        static glm::vec3 v2 = { -unit,  unit, 0.0f };

        ManipulatePosition(camera, &v0, 1);
        ManipulatePosition(camera, &v1, 1);
        ManipulatePosition(camera, &v2, 1);

        DrawText(v0, "v0");
        DrawText(v1, "v1");
        DrawText(v2, "v2");
        DrawLine(v0, v1, { 128 , 128 , 128 });
        DrawLine(v1, v2, { 128 , 128 , 128 });
        DrawLine(v2, v0, { 128 , 128 , 128 });

        static bool sixSeparating = true;
        static float dps = 0.1f;
        static glm::vec3 origin = { -2.0f, -2.0f, -2.0f };
        static int gridRes = 32;

        DrawText(origin, "origin");
        ManipulatePosition( camera, &origin, 1 );
        DrawAABB(origin, origin + glm::vec3(dps, dps, dps) * (float)gridRes, { 255 ,0 ,0 });

        VoxelTriangleIntersector intersector( v0, v1, v2, sixSeparating, dps );
        
#if 0
        {
            glm::ivec3 lower = glm::ivec3( glm::floor( (intersector.triangle_lower - origin ) / dps ) );
            glm::ivec3 upper = glm::ivec3( glm::floor( (intersector.triangle_upper - origin ) / dps ) );
            lower = glm::max( lower, glm::ivec3( 0, 0, 0 ) );
            upper = glm::min( upper, glm::ivec3( gridRes-1, gridRes-1, gridRes-1 ) );

            for( int x = lower.x; x <= upper.x ; x++ )
            for( int y = lower.y; y <= upper.y ; y++ )
            for( int z = lower.z; z <= upper.z ; z++ )
            {
                glm::vec3 p = origin + glm::vec3( x, y, z ) * dps;
                bool overlap = intersector.intersect( p, dps );
                if (overlap)
                {
                    DrawAABB(p, p + glm::vec3(dps, dps, dps), {200 ,200 ,200});
                }
            }
        }
#else
        //static glm::vec3 projp = { 0.0f, 0.0f, 0.0f };

        //projp.z = origin.z;

        //DrawText(projp, "projp");
        //ManipulatePosition(camera, &projp, 1);

        // float t = glm::dot(v0 - projp, intersector.n) / intersector.n.z;
        // DrawSphere({ projp.x, projp.y, projp.z + t }, 0.02f, { 255,255,0 });
        
        VoxelTriangleVisitor visitor(origin, intersector.triangle_lower, intersector.triangle_upper, v0, intersector.n, dps, gridRes );
        for (int x = visitor.lower_xy.x; x <= visitor.upper_xy.x; x++)
        {
            glm::ivec2 yrange = visitor.yRangeInclusive( intersector, x, dps );
            for (int y = yrange.x; y <= yrange.y ; y++ )
            {
                glm::ivec2 zrange = visitor.zRangeInclusive(x, y, dps, gridRes, sixSeparating );
                for (int z = zrange.x; z <= zrange.y; z++)
                {
                    glm::vec3 p = visitor.p(x, y, z, dps);
                    bool overlap = intersector.intersect(p, dps );
                    if (overlap)
                    {
                        DrawAABB(p, p + glm::vec3(dps, dps, dps), { 200 ,200 ,200 });
                    }
                }
            }
        }
#endif
#endif

        //glm::vec3 e01 = v1 - v0;
        //glm::vec3 e12 = v2 - v1;
        //glm::vec3 n = glm::cross( e01, e12 );

        //float dps = 1.0f;
        //glm::vec3 dp = glm::vec3(dps, dps, dps);

        //glm::vec3 triangle_lower = glm::min(glm::min(v0, v1), v2);
        //glm::vec3 triangle_upper = glm::max(glm::max(v0, v1), v2);

        //{
        //    glm::vec3 c = glm::vec3(
        //        0.0f < n.x ? dps : 0.0f,
        //        0.0f < n.y ? dps : 0.0f,
        //        0.0f < n.z ? dps : 0.0f
        //    );

        //    float d1;
        //    float d2;

        //    if (sixSeparating == false)
        //    {
        //        d1 = glm::dot(n, c - v0);
        //        d2 = glm::dot(n, dp - c - v0);
        //    }
        //    else
        //    {
        //        float k1 = glm::dot(n, dp * 0.5f - v0);
        //        float k2 = 0.5f * dps * glm::max( glm::max( glm::abs(n.x), glm::abs(n.y) ), glm::abs(n.z) );
        //        d1 = k1 - k2;
        //        d2 = k1 + k2;
        //    }

        //    float d_consts[3 /*axis*/][3 /*edge*/];
        //    glm::vec2 nes[3 /*axis*/][3 /*edge*/];
        //    for (int axis = 0; axis < 3 ; axis++)
        //    {
        //        glm::vec2 dp_proj = glm::vec2(dps, dps);
        //        glm::vec2 vs_proj[3] = {
        //            project2plane(v0, axis),
        //            project2plane(v1, axis),
        //            project2plane(v2, axis),
        //        };
        //        float reminder = project2plane_reminder(n, axis);

        //        for (int edge = 0; edge < 3; edge++)
        //        {
        //            glm::vec2 a = vs_proj[edge];
        //            glm::vec2 b = vs_proj[(edge + 1) % 3];
        //            glm::vec2 e = b - a;
        //            glm::vec2 ne = glm::vec2(-e.y, e.x);
        //            if (reminder < 0.0f) {
        //                ne = -ne;
        //            }
        //            nes[axis][edge] = ne;

        //            float d_const;
        //            if (sixSeparating == false)
        //            {
        //                d_const = glm::max(ne.x * dp_proj.x, 0.0f)
        //                    + glm::max(ne.y * dp_proj.y, 0.0f)
        //                    - glm::dot(ne, a);
        //            }
        //            else
        //            {
        //                d_const = glm::dot(ne, dp_proj * 0.5f - a)
        //                    + 0.5f * dps * glm::max(glm::abs(ne.x), glm::abs(ne.y));
        //            }
        //            d_consts[axis][edge] = d_const;
        //        }
        //    }

        //    for( int x = -25; x < 25; x++ )
        //    for( int y = -25; y < 25; y++ )
        //    for( int z = -25; z < 25; z++ )
        //    {
        //        glm::vec3 p = glm::vec3( x, y, z );

        //        float PoN = glm::dot( p, n );
        //        if (0.0f < (PoN + d1) * (PoN + d2))
        //        {
        //            continue;
        //        }

        //        //DrawAABB( p, p + dp, { 200 ,200 ,200 });

        //        bool overlap = true;

        //        // bbox test for a corner case
        //        if (overlapAABB(p, p + dp, triangle_lower, triangle_upper) == false )
        //        {
        //            overlap = false;
        //        }

        //        // projection test
        //        for( int axis = 0; axis < 3 && overlap ; axis++ )
        //        {
        //            glm::vec2 p_proj = project2plane(p, axis);
        //            for( int edge = 0; edge < 3 && overlap; edge++ )
        //            {
        //                float d = glm::dot(nes[axis][edge], p_proj) + d_consts[axis][edge];
        //                if (d < 0.0f)
        //                {
        //                    overlap = false;
        //                }
        //            }
        //        }

        //        if (overlap)
        //        {
        //            DrawAABB(p, p + dp, { 200 ,200 ,200 });
        //        }
        //    }
        //}

        //for( int x = -5; x < 5; x++ )
        //for( int y = -5; y < 5; y++ )
        //{
        //    glm::vec3 p = glm::vec3(x, y, 4);

        //    glm::vec2 p_proj  = project2plane(p, 0);
        //    glm::vec2 dp_proj = glm::vec2(dps, dps);
        //    glm::vec2 vs_proj[3] = {
        //        project2plane( v0, 0 ),
        //        project2plane( v1, 0 ),
        //        project2plane( v2, 0 ),
        //    };

        //    // bbox
        //    // glm::vec2

        //    float reminder = project2plane_reminder( n, 0 );
        //    bool overlap = true;
        //    for (int i = 0; i < 3; i++)
        //    {
        //        glm::vec2 a = vs_proj[i];
        //        glm::vec2 b = vs_proj[(i + 1) % 3];
        //        glm::vec2 e = b - a;
        //        glm::vec2 ne = glm::vec2( -e.y, e.x );
        //        if (reminder < 0.0f ) {
        //            ne = -ne;
        //        }
        //        float d = glm::dot(ne, p_proj)
        //            + glm::max(ne.x * dp_proj.x, 0.0f)
        //            + glm::max(ne.y * dp_proj.y, 0.0f)
        //            - glm::dot(ne, a);

        //        if (d < 0.0f)
        //        {
        //            overlap = false;
        //            break;
        //        }

        //        DrawLine({ a, 4 }, { b, 4 }, { 255 , 128 , 128 });
        //    }

        //    if (overlap)
        //    {
        //        DrawAABB(p, p + dp, { 200 ,200 ,200 });
        //    }
        //}

        //for( int x = -5; x < 5; x++ )
        //for( int y = -5; y < 5; y++ )
        //{
        //    glm::vec3 p = glm::vec3(x, y, 4);

        //    glm::vec2 p_proj  = project2plane(p, 0);
        //    glm::vec2 dp_proj = glm::vec2(dps, dps);
        //    glm::vec2 vs_proj[3] = {
        //        project2plane( v0, 0 ),
        //        project2plane( v1, 0 ),
        //        project2plane( v2, 0 ),
        //    };

        //    // bbox
        //    // glm::vec2

        //    float reminder = project2plane_reminder( n, 0 );
        //    bool overlap = true;
        //    for (int i = 0; i < 3; i++)
        //    {
        //        glm::vec2 a = vs_proj[i];
        //        glm::vec2 b = vs_proj[(i + 1) % 3];
        //        glm::vec2 e = b - a;
        //        glm::vec2 ne = glm::vec2( -e.y, e.x );
        //        if (reminder < 0.0f ) {
        //            ne = -ne;
        //        }

        //        int major = majorAxis( ne );
        //        float d = glm::dot(ne, p_proj)
        //            + glm::dot(ne, dp_proj * 0.5f - a)
        //            + 0.5f * dp_proj[major] * glm::abs( ne[major] );

        //        if (d < 0.0f)
        //        {
        //            overlap = false;
        //            break;
        //        }

        //        DrawLine({ a, 4 }, { b, 4 }, { 255 , 128 , 128 });
        //    }

        //    if (overlap)
        //    {
        //        DrawAABB(p, p + dp, { 200 ,200 ,200 });
        //    }
        //}

        PopGraphicState();
        EndCamera();

        BeginImGui();

        ImGui::SetNextWindowSize({ 500, 800 }, ImGuiCond_Once);
        ImGui::Begin("Panel");
        ImGui::Text("fps = %f", GetFrameRate());
        ImGui::InputFloat("dps", &dps, 0.01f);
        ImGui::InputInt("gridRes", &gridRes);
        ImGui::Checkbox("sixSeparating", &sixSeparating);
        //ImGui::Checkbox("planeVoxelTest", &planeVoxelTest);
        ImGui::Text("voxel: %f s", voxel_time);
        
        ImGui::End();

        EndImGui();
    }

    pr::CleanUp();
}
