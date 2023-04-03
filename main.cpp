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

int majorAxis( glm::vec3 d )
{
    float x = glm::abs(d.x);
    float y = glm::abs(d.y);
    float z = glm::abs(d.z);
    if (x < y)
    {
        return y < z ? 2 : 1;
    }
    return x < z ? 2 : 0;
}
int majorAxis( glm::vec2 d )
{
    float x = glm::abs(d.x);
    float y = glm::abs(d.y);
    return x < y ? 1 : 0;
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
            UpdateCameraBlenderLike(&camera  );
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

        static bool sixSeparating = true;

        glm::vec3 e01 = v1 - v0;
        glm::vec3 e12 = v2 - v1;
        glm::vec3 n = ( glm::cross( e01, e12 ) );

        glm::vec3 dp = glm::vec3(1.0f, 1.0f, 1.0f);

        glm::vec3 triangle_lower = glm::min(glm::min(v0, v1), v2);
        glm::vec3 triangle_upper = glm::max(glm::max(v0, v1), v2);

        
        {
            glm::vec3 c = glm::vec3(
                0.0f < n.x ? dp.x : 0.0f,
                0.0f < n.y ? dp.y : 0.0f,
                0.0f < n.z ? dp.z : 0.0f
            );

            float d1;
            float d2;

            if (sixSeparating == false)
            {
                d1 = glm::dot(n, c - v0);
                d2 = glm::dot(n, dp - c - v0);
            }
            else
            {
                int major = majorAxis(n);
                float k1 = glm::dot(n, dp * 0.5f - v0);
                float k2 = 0.5f * n[major] * dp[major];
                d1 = k1 - k2;
                d2 = k1 + k2;
            }

            float d_consts[3 /*axis*/][3 /*edge*/];
            glm::vec2 nes[3 /*axis*/][3 /*edge*/];
            for (int axis = 0; axis < 3 ; axis++)
            {
                glm::vec2 dp_proj = project2plane(dp, axis);
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
                        int major = majorAxis(ne);
                        d_const = glm::dot(ne, dp_proj * 0.5f - a)
                            + 0.5f * dp_proj[major] * glm::abs(ne[major]);
                    }
                    d_consts[axis][edge] = d_const;
                }
            }

            for( int x = -25; x < 25; x++ )
            for( int y = -25; y < 25; y++ )
            for( int z = -25; z < 25; z++ )
            {
                glm::vec3 p = glm::vec3( x, y, z );

                float PoN = glm::dot( p, n );
                if (0.0f < (PoN + d1) * (PoN + d2))
                {
                    continue;
                }

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
                    for( int edge = 0; edge < 3 && overlap; edge++ )
                    {
                        float d = glm::dot(nes[axis][edge], p_proj) + d_consts[axis][edge];
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
        ImGui::Checkbox("sixSeparating", &sixSeparating);
        ImGui::End();

        EndImGui();
    }

    pr::CleanUp();
}
