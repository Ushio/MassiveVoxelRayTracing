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

	const char* input = "bunny.obj";
	SetDataDir( ExecutableDir() );
	std::string errorMsg;
	std::shared_ptr<FScene> scene = ReadWavefrontObj( GetDataPath( input ), errorMsg );

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

        //static glm::vec3 octree_lower;
        //static glm::vec3 octree_upper;

        //scene->visitPolyMesh([](std::shared_ptr<const FPolyMeshEntity> polymesh) {
        //    ColumnView<int32_t> faceCounts(polymesh->faceCounts());
        //    ColumnView<int32_t> indices(polymesh->faceIndices());
        //    ColumnView<glm::vec3> positions(polymesh->positions());

        //    // Geometry
        //    pr::PrimBegin(pr::PrimitiveMode::Lines);
        //    for (int i = 0; i < positions.count(); i++)
        //    {
        //        glm::vec3 p = positions[i];
        //        glm::ivec3 color = { 255,255,255 };
        //        pr::PrimVertex(p, { color });
        //    }
        //    int indexBase = 0;

        //    for (int i = 0; i < faceCounts.count(); i++)
        //    {
        //        int nVerts = faceCounts[i];
        //        for (int j = 0; j < nVerts; ++j)
        //        {
        //            int i0 = indices[indexBase + j];
        //            int i1 = indices[indexBase + (j + 1) % nVerts];
        //            pr::PrimIndex(i0);
        //            pr::PrimIndex(i1);
        //        }
        //        indexBase += nVerts;
        //    }
        //    pr::PrimEnd();

        //    // Assume Triangle

        //    glm::vec3 lower = glm::vec3(FLT_MAX, FLT_MAX, FLT_MAX);
        //    glm::vec3 upper = glm::vec3(-FLT_MAX, -FLT_MAX, -FLT_MAX);

        //    for (int i = 0; i < faceCounts.count(); i++)
        //    {
        //        for (int j = 0; j < 3; ++j)
        //        {
        //            int index = indices[i * 3 + j];
        //            lower = glm::min(lower, positions[index]);
        //            upper = glm::max(upper, positions[index]);
        //        }
        //    }

        //    // bounding box
        //    glm::vec3 size = upper - lower;
        //    float dps = glm::max(glm::max(size.x, size.y), size.z) / (float)gridRes;

        //    octree_lower = lower;
        //    octree_upper = lower + glm::vec3(dps, dps, dps) * (float)gridRes;

        //    DrawAABB(lower, lower + glm::vec3(dps, dps, dps) * (float)gridRes, { 255 ,0 ,0 });

        //    std::set<uint64_t> mortonVoxels;

        //    Stopwatch voxelsw;

        //    glm::vec3 origin = lower;

        //    for (int i = 0; i < faceCounts.count(); i++)
        //    {
        //        glm::vec3 v0 = positions[indices[i * 3]];
        //        glm::vec3 v1 = positions[indices[i * 3 + 1]];
        //        glm::vec3 v2 = positions[indices[i * 3 + 2]];

        //        VTContext context(v0, v1, v2, sixSeparating, origin, dps, gridRes);
        //        glm::ivec2 xrange = context.xRangeInclusive();
        //        for (int x = xrange.x; x <= xrange.y; x++)
        //        {
        //            glm::ivec2 yrange = context.yRangeInclusive(x, dps);
        //            for (int y = yrange.x; y <= yrange.y; y++)
        //            {
        //                glm::ivec2 zrange = context.zRangeInclusive(x, y, dps, sixSeparating);
        //                for (int z = zrange.x; z <= zrange.y; z++)
        //                {
        //                    glm::vec3 p = context.p(x, y, z, dps);
        //                    if (context.intersect(p))
        //                    {
        //                        glm::ivec3 c = context.i(x, y, z);
        //                        mortonVoxels.insert(encode2mortonCode_PDEP(c.x, c.y, c.z));
        //                    }
        //                }
        //            }
        //        }
        //    } // face

        //    voxel_time = voxelsw.elapsed();

        //    // Draw
        //    if( drawVoxelWire )
        //    {
        //        //for (auto morton : mortonVoxels)
        //        //{
        //        //    glm::uvec3 c;
        //        //    decodeMortonCode_PEXT(morton, &c.x, &c.y, &c.z);
        //        //    glm::vec3 p = origin + glm::vec3(c.x, c.y, c.z) * dps;
        //        //    DrawAABB(p, p + glm::vec3(dps, dps, dps), { 200 ,200 ,200 });
        //        //}
        //        drawVoxels(mortonVoxels, origin, dps, { 200 ,200 ,200 });
        //    }

        //    // voxel build
        //    buildOctree( &nodes, mortonVoxels, gridRes );

        //    embreeVoxel = std::shared_ptr<EmbreeVoxel>(new EmbreeVoxel(mortonVoxels, octree_lower, octree_upper, gridRes));
        //});

		PopGraphicState();
		EndCamera();

		BeginImGui();

		ImGui::SetNextWindowSize( { 500, 800 }, ImGuiCond_Once );
		ImGui::Begin( "Panel" );
		ImGui::Text( "fps = %f", GetFrameRate() );


		ImGui::End();

		EndImGui();
	}

	pr::CleanUp();
}