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

    while (pr::NextFrame() == false) {
        if (IsImGuiUsingMouse() == false) {
            UpdateCameraBlenderLike(&camera);
        }

        ClearBackground(0.1f, 0.1f, 0.1f, 1);

        BeginCamera(camera);

        PushGraphicState();

        DrawGrid(GridAxis::XZ, 1.0f, 10, { 128, 128, 128 });
        DrawXYZAxis(1.0f);

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

        glm::vec3 e01 = v1 - v0;
        glm::vec3 e12 = v2 - v1;
        glm::vec3 n = ( glm::cross( e01, e12 ) );

        glm::vec3 dp = glm::vec3(1.0f, 1.0f, 1.0f);
        glm::vec3 c = glm::vec3(
            0.0f < n.x ? dp.x : 0.0f,
            0.0f < n.y ? dp.y : 0.0f,
            0.0f < n.z ? dp.z : 0.0f
        );

        float eps = FLT_EPSILON;
        float d1 = glm::dot(n, c - v0);
        float d2 = glm::dot(n, dp - c - v0);

        glm::vec3 triangle_lower = glm::min( glm::min( v0, v1 ), v2 );
        glm::vec3 triangle_upper = glm::max( glm::max( v0, v1 ), v2 );

        for( int x = -15; x < 15; x++ )
        for( int y = -15; y < 15; y++ )
        for( int z = -15; z < 15; z++ )
        {
            glm::vec3 p = glm::vec3( x, y, z );

            float PoN = glm::dot( p, n );
            if( ( PoN + d1 ) * ( PoN + d2 ) < eps)
            {
                // DrawAABB( p, p + dp, { 200 ,200 ,200 });

                bool overlap = true;

                // bbox test for a corner case
                if (overlapAABB(p, p + dp, triangle_lower, triangle_upper) == false )
                {
                    overlap = false;
                }

                // projection test
                for( int axis = 0; axis < 3 && overlap ; axis++ )
                {
                    glm::vec2 p_proj = project2plane(p, axis);
                    glm::vec2 dp_proj = project2plane(dp, axis);
                    glm::vec2 vs_proj[3] = {
                        project2plane(v0, axis),
                        project2plane(v1, axis),
                        project2plane(v2, axis),
                    };
                    float reminder = project2plane_reminder(n, axis);
                    
                    for( int i = 0; i < 3 && overlap; i++ )
                    {
                        glm::vec2 a = vs_proj[i];
                        glm::vec2 b = vs_proj[(i + 1) % 3];
                        glm::vec2 e = b - a;
                        glm::vec2 ne = glm::vec2(-e.y, e.x);
                        if (reminder < 0.0f) {
                            ne = -ne;
                        }
                        float d = glm::dot(ne, p_proj)
                            + glm::max(ne.x * dp_proj.x, 0.0f)
                            + glm::max(ne.y * dp_proj.y, 0.0f)
                            - glm::dot(ne, a);

                        if (d < 0.0f)
                        {
                            overlap = false;
                        }
                    }
                }

                if (overlap)
                {
                    DrawAABB(p, p + dp, { 200 ,200 ,200 });
                }
            }
        }

        //for( int x = -5; x < 5; x++ )
        //for( int y = -5; y < 5; y++ )
        //{
        //    glm::vec3 p = glm::vec3(x, y, 4);

        //    glm::vec2 p_proj  = project2plane(p, 0);
        //    glm::vec2 dp_proj = project2plane(dp, 0);
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

        PopGraphicState();
        EndCamera();

        BeginImGui();

        ImGui::SetNextWindowSize({ 500, 800 }, ImGuiCond_Once);
        ImGui::Begin("Panel");
        ImGui::Text("fps = %f", GetFrameRate());

        ImGui::End();

        EndImGui();
    }

    pr::CleanUp();
}
