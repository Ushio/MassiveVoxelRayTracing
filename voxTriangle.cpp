#include "pr.hpp"
#include "voxelization.hpp"
#include <iostream>
#include <memory>
#include <set>

int main()
{
	using namespace pr;

	Config config;
	config.ScreenWidth = 1920;
	config.ScreenHeight = 1080;
	config.SwapInterval = 0;
	Initialize( config );

	Camera3D camera;
	camera.origin = { 4, 4, 4 };
	camera.lookat = { 0, 0, 0 };
	camera.zUp = false;

	double e = GetElapsedTime();

	SetDataDir( ExecutableDir() );

	while( pr::NextFrame() == false )
	{
		if( IsImGuiUsingMouse() == false )
		{
			UpdateCameraBlenderLike( &camera );
		}
		ClearBackground( 0.1f, 0.1f, 0.1f, 1 );

		BeginCamera( camera );

		PushGraphicState();

		DrawGrid( GridAxis::XZ, 1.0f, 10, { 128, 128, 128 } );
		DrawXYZAxis( 1.0f );

		float unit = 1.0f;
		static glm::vec3 v0 = { -unit, -unit - 0.3f, 0.0f };
		static glm::vec3 v1 = { unit, -unit, 0.0f };
		static glm::vec3 v2 = { -unit, unit, 0.0f };

		ManipulatePosition( camera, &v0, 1 );
		ManipulatePosition( camera, &v1, 1 );
		ManipulatePosition( camera, &v2, 1 );

		DrawText( v0, "v0" );
		DrawText( v1, "v1" );
		DrawText( v2, "v2" );
		DrawLine( v0, v1, { 128, 128, 128 } );
		DrawLine( v1, v2, { 128, 128, 128 } );
		DrawLine( v2, v0, { 128, 128, 128 } );

		static bool sixSeparating = true;
		static float dps = 0.1f;
		static glm::vec3 origin = { -2.0f, -2.0f, -2.0f };
		static int gridRes = 32;

		DrawText( origin, "origin" );
		ManipulatePosition( camera, &origin, 1 );
		DrawAABB( origin, origin + glm::vec3( dps, dps, dps ) * (float)gridRes, { 255, 0, 0 } );

		VTContext context( v0, v1, v2, sixSeparating, origin, dps, gridRes );
		glm::ivec2 xrange = context.xRangeInclusive();
		for( int x = xrange.x; x <= xrange.y; x++ )
		{
			glm::ivec2 yrange = context.yRangeInclusive( x, dps );
			for( int y = yrange.x; y <= yrange.y; y++ )
			{
				glm::ivec2 zrange = context.zRangeInclusive( x, y, dps, sixSeparating );
				for( int z = zrange.x; z <= zrange.y; z++ )
				{
					glm::vec3 p = context.p( x, y, z, dps );
					if( context.intersect( p ) )
					{
						DrawAABB( p, p + glm::vec3( dps, dps, dps ), { 200, 200, 200 } );
					}
				}
			}
		}

		PopGraphicState();
		EndCamera();

		BeginImGui();

		ImGui::SetNextWindowSize( { 500, 800 }, ImGuiCond_Once );
		ImGui::Begin( "Panel" );
		ImGui::Text( "fps = %f", GetFrameRate() );
		
		ImGui::Checkbox( "sixSeparating", &sixSeparating );
		int prevGridRes = gridRes;
		if (ImGui::InputInt("gridRes", &gridRes))
		{
			dps *= (float)prevGridRes / gridRes;
		}
		ImGui::InputFloat( "dps", &dps, 0.01f );

		ImGui::End();

		EndImGui();
	}

	pr::CleanUp();
}