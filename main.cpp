#include "pr.hpp"
#include <iostream>
#include <memory>

glm::vec2 project2plane(glm::vec3 p, int axis)
{
    glm::vec2 r;
    switch (axis)
    {
    case 0:
        r.x = p.x;
        r.y = p.y;
        break;
    case 1:
        r.x = p.y;
        r.y = p.z;
        break;
    case 2:
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

struct VoxelTriangleIntersector
{
    glm::vec3 n;
    glm::vec3 triangle_lower;
    glm::vec3 triangle_upper;
    float d1;
    float d2;
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

        if (sixSeparating == false)
        {
            d1 = glm::dot(n, c - v0);
            d2 = glm::dot(n, dp - c - v0);
        }
        else
        {
            float k1 = glm::dot(n, dp * 0.5f - v0);
            float k2 = 0.5f * dps * glm::max(glm::max(glm::abs(n.x), glm::abs(n.y)), glm::abs(n.z));
            d1 = k1 - k2;
            d2 = k1 + k2;
        }

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
        float PoN = glm::dot(p, n);
        if (0.0f < (PoN + d1) * (PoN + d2))
        {
            return false;
        }

        bool overlap = true;

        // bbox test for a corner case
        if (overlapAABB(p, p + glm::vec3( dps, dps, dps ), triangle_lower, triangle_upper) == false)
        {
            return false;
        }

        // projection test
        for (int axis = 0; axis < 3 && overlap; axis++)
        {
            glm::vec2 p_proj = project2plane(p, axis);
            for (int edge = 0; edge < 3 && overlap; edge++)
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

    SetDataDir(ExecutableDir());
    std::string errorMsg;
    std::shared_ptr<FScene> scene = ReadWavefrontObj( GetDataPath("Tri.obj"), errorMsg );

    while (pr::NextFrame() == false) {
        if (IsImGuiUsingMouse() == false) {
            UpdateCameraBlenderLike(&camera  );
        }

        ClearBackground(0.1f, 0.1f, 0.1f, 1);

        BeginCamera(camera);

        PushGraphicState();

        DrawGrid(GridAxis::XZ, 1.0f, 10, { 128, 128, 128 });
        DrawXYZAxis(1.0f);

#if 0
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
            glm::vec3 origin = lower;

            for (int i = 0; i < faceCounts.count(); i++)
            {
                glm::vec3 v0 = positions[indices[i * 3]];
                glm::vec3 v1 = positions[indices[i * 3 + 1]];
                glm::vec3 v2 = positions[indices[i * 3 + 2]];

                VoxelTriangleIntersector intersector(v0, v1, v2, sixSeparating, dps);

                glm::vec3 conservativeLower = intersector.triangle_lower - glm::vec3(dps, dps, dps) / 128.0f;
                glm::vec3 conservativeUpper = intersector.triangle_upper + glm::vec3(dps, dps, dps) / 128.0f;

                glm::ivec3 lower = glm::ivec3(glm::floor((conservativeLower - origin) / dps));
                glm::ivec3 upper = glm::ivec3(glm::floor((conservativeUpper - origin) / dps));

                lower = glm::max(lower, glm::ivec3(0, 0, 0));
                upper = glm::max(upper, glm::ivec3(0, 0, 0));
                lower = glm::min(lower, glm::ivec3(gridRes - 1, gridRes - 1, gridRes - 1));
                upper = glm::min(upper, glm::ivec3(gridRes - 1, gridRes - 1, gridRes - 1));

                for (int x = lower.x; x <= upper.x; x++)
                for (int y = lower.y; y <= upper.y; y++)
                for (int z = lower.z; z <= upper.z; z++)
                {
                    glm::vec3 p = origin + glm::vec3(x, y, z) * dps;
                    bool overlap = intersector.intersect(p, dps);
                    if (overlap)
                    {
                        DrawAABB(p, p + glm::vec3(dps, dps, dps), { 200 ,200 ,200 });
                    }
                }
            }
        });
#endif

#if 1
        float unit = 1.0f;
        static glm::vec3 v0 = { -unit, -unit, 0.0f };
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

        glm::vec3 conservativeLower = intersector.triangle_lower - glm::vec3( dps, dps, dps ) / 128.0f;
        glm::vec3 conservativeUpper = intersector.triangle_upper + glm::vec3( dps, dps, dps ) / 128.0f;

        glm::ivec3 lower = glm::ivec3( glm::floor( ( conservativeLower - origin ) / dps ) );
        glm::ivec3 upper = glm::ivec3( glm::floor( ( conservativeUpper - origin ) / dps ) );

        lower = glm::max( lower, glm::ivec3( 0, 0, 0 ) );
        upper = glm::max( upper, glm::ivec3( 0, 0, 0 ) );
        lower = glm::min( lower, glm::ivec3( gridRes-1, gridRes-1, gridRes-1 ) );
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
        ImGui::End();

        EndImGui();
    }

    pr::CleanUp();
}
