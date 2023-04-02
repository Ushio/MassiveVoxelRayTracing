#include "pr.hpp"
#include <iostream>
#include <memory>

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

        float unit = 3.0f;
        static glm::vec3 p0 = { -unit, -unit, 0.0f };
        static glm::vec3 p1 = { unit , -unit, 0.0f };
        static glm::vec3 p2 = { -unit,  unit, 0.0f };

        ManipulatePosition(camera, &p0, 0.3f);
        ManipulatePosition(camera, &p1, 0.3f);
        ManipulatePosition(camera, &p2, 0.3f);

        DrawText(p0, "p0");
        DrawText(p1, "p1");
        DrawText(p2, "p2");
        DrawLine(p0, p1, { 128 , 128 , 128 });
        DrawLine(p1, p2, { 128 , 128 , 128 });
        DrawLine(p2, p0, { 128 , 128 , 128 });

        glm::vec3 e01 = p1 - p0;
        glm::vec3 e12 = p2 - p1;
        glm::vec3 n = ( glm::cross( e01, e12 ) );

        glm::vec3 dp = glm::vec3(1.0f, 1.0f, 1.0f);
        glm::vec3 c = glm::vec3(
            0.0f < n.x ? dp.x : 0.0f,
            0.0f < n.y ? dp.y : 0.0f,
            0.0f < n.z ? dp.z : 0.0f
        );
        glm::vec3 v0 = p0;

        float eps = FLT_EPSILON;
        float d1 = glm::dot(n, c - v0);
        float d2 = glm::dot(n, dp - c - v0);

        for( int x = -5; x < 5; x++ )
        for( int y = -5; y < 5; y++ )
        for( int z = -5; z < 5; z++ )
        {
            glm::vec3 p = glm::vec3( x, y, z );
            float PoN = glm::dot( p, n );
            if( ( PoN + d1 ) * ( PoN + d2 ) < eps)
            {
                DrawAABB( p, p + dp, { 200 ,200 ,200 });
            }
        }


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
