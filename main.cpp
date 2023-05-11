﻿#include "pr.hpp"
#include <iostream>
#include <set>
#include <memory>
#include "voxelization.hpp"

//bool overlapAABB( glm::vec3 lowerA, glm::vec3 upperA, glm::vec3 lowerB, glm::vec3 upperB)
//{
//    if (upperA.x < lowerB.x || upperA.y < lowerB.y || upperA.z < lowerB.z)
//    {
//        return false;
//    }
//
//    if (upperB.x < lowerA.x || upperB.y < lowerA.y || upperB.z < lowerA.z)
//    {
//        return false;
//    }
//    return true;
//}

// morton max 0x1FFFFF

inline uint64_t encode2mortonCode( uint32_t x, uint32_t y, uint32_t z ) 
{
    uint64_t code = 0;
    for (uint64_t i = 0; i < 64 / 3; ++i ) {
        code |= 
            ( (uint64_t)(x & (1u << i)) << (2 * i + 0)) |
            ( (uint64_t)(y & (1u << i)) << (2 * i + 1)) | 
            ( (uint64_t)(z & (1u << i)) << (2 * i + 2));
    }
    return code;
}

inline void decodeMortonCode( uint64_t morton, uint32_t* x, uint32_t* y, uint32_t* z )
{
    uint32_t ox = 0;
    uint32_t oy = 0;
    uint32_t oz = 0;
    for( uint64_t i = 0; i < 64 / 3; ++i )
    {
        uint64_t a = morton & 0x1;
        uint64_t b = ( morton & 0x2 ) >> 1;
        uint64_t c = ( morton & 0x4 ) >> 2;
        morton = morton >> 3;
        ox |= a << i;
        oy |= b << i;
        oz |= c << i;
    }
    *x = ox;
    *y = oy;
    *z = oz;
}

/*
* https://www.chessprogramming.org/BMI2
SRC1   ┌───┬───┬───┬───┬───┐    ┌───┬───┬───┬───┬───┬───┬───┬───┐
       │S63│S62│S61│S60│S59│....│ S7│ S6│ S5│ S4│ S3│ S2│ S1│ S0│
       └───┴───┴───┴───┴───┘    └───┴───┴───┴───┴───┴───┴───┴───┘

SRC2   ┌───┬───┬───┬───┬───┐    ┌───┬───┬───┬───┬───┬───┬───┬───┐
(mask) │ 0 │ 0 │ 0 │ 1 │ 0 │0...│ 1 │ 0 │ 1 │ 0 │ 0 │ 1 │ 0 │ 0 │  (f.i. 4 bits set)
       └───┴───┴───┴───┴───┘    └───┴───┴───┴───┴───┴───┴───┴───┘

DEST   ┌───┬───┬───┬───┬───┐    ┌───┬───┬───┬───┬───┬───┬───┬───┐
       │ 0 │ 0 │ 0 │ S3│ 0 │0...│ S2│ 0 │ S1│ 0 │ 0 │ S0│ 0 │ 0 │
       └───┴───┴───┴───┴───┘    └───┴───┴───┴───┴───┴───┴───┴───┘
*/


inline uint64_t encode2mortonCode_PDEP( uint32_t x, uint32_t y, uint32_t z )
{
    uint64_t code = 
        _pdep_u64(x & 0x1FFFFF, 0x1249249249249249LLU) |
        _pdep_u64(y & 0x1FFFFF, 0x1249249249249249LLU << 1) |
        _pdep_u64(z & 0x1FFFFF, 0x1249249249249249LLU << 2);
    return code;
}

// method to seperate bits from a given integer 3 positions apart
inline uint64_t splitBy3( uint32_t a) {
    uint64_t x = a & 0x1FFFFF;
    x = (x | x << 32) & 0x1f00000000ffff; // shift left 32 bits, OR with self, and 00011111000000000000000000000000000000001111111111111111
    x = (x | x << 16) & 0x1f0000ff0000ff; // shift left 32 bits, OR with self, and 00011111000000000000000011111111000000000000000011111111
    x = (x | x << 8) & 0x100f00f00f00f00f; // shift left 32 bits, OR with self, and 0001000000001111000000001111000000001111000000001111000000000000
    x = (x | x << 4) & 0x10c30c30c30c30c3; // shift left 32 bits, OR with self, and 0001000011000011000011000011000011000011000011000011000100000000
    x = (x | x << 2) & 0x1249249249249249;
    return x;
}
inline uint64_t encode2mortonCode_magicbits(uint32_t x, uint32_t y, uint32_t z) {
    uint64_t answer = 0;
    answer |= splitBy3(x) | splitBy3(y) << 1 | splitBy3(z) << 2;
    return answer;
}

static bool experiment = true;

class VoxelObjWriter
{
public:
    void add( glm::vec3 p, float dps)
    {
        points.push_back(p);
        points.push_back(p + glm::vec3(dps, 0, 0));
        points.push_back(p + glm::vec3(dps, 0, dps));
        points.push_back(p + glm::vec3(0, 0, dps));
        points.push_back(p + glm::vec3(0, dps, 0));
        points.push_back(p + glm::vec3(dps, dps, 0));
        points.push_back(p + glm::vec3(dps, dps, dps));
        points.push_back(p + glm::vec3(0, dps, dps));
    }
    void saveObj( const char* file )
    {
        FILE* fp = fopen( file, "w" );
        for (int i = 0; i < points.size(); i++)
        {
            fprintf( fp, "v %.6f %.6f %.6f\n", points[i].x, points[i].y, points[i].z );
        }
        int nVoxels = points.size() / 8;
        for (int i = 0; i < nVoxels; i++)
        {
            uint32_t i0 = i * 8 + 1;
            uint32_t i1 = i * 8 + 2;
            uint32_t i2 = i * 8 + 3;
            uint32_t i3 = i * 8 + 4;
            uint32_t i4 = i * 8 + 5;
            uint32_t i5 = i * 8 + 6;
            uint32_t i6 = i * 8 + 7;
            uint32_t i7 = i * 8 + 8;

            // Left Hand
            fprintf(fp, "f %d %d %d %d\n", i0, i1, i2, i3);
            fprintf(fp, "f %d %d %d %d\n", i7, i6, i5, i4);
            fprintf(fp, "f %d %d %d %d\n", i4, i5, i1, i0);
            fprintf(fp, "f %d %d %d %d\n", i5, i6, i2, i1);
            fprintf(fp, "f %d %d %d %d\n", i6, i7, i3, i2);
            fprintf(fp, "f %d %d %d %d\n", i7, i4, i0, i3);
        }
        fclose( fp );
    }

    void savePLY(const char* file)
    {
        FILE* fp = fopen(file, "wb");

        int nVoxels = points.size() / 8;

        // PLY header
        fprintf(fp, "ply\n");
        fprintf(fp, "format binary_little_endian 1.0\n");
        fprintf(fp, "element vertex %llu\n", points.size() );
        fprintf(fp, "property float x\n");
        fprintf(fp, "property float y\n");
        fprintf(fp, "property float z\n");
        fprintf(fp, "element face %d\n", nVoxels * 6 );
        fprintf(fp, "property list uchar uint vertex_indices\n");
        fprintf(fp, "end_header\n");

        // Write vertices
        fwrite(points.data(), sizeof( glm::vec3 ) * points.size(), 1, fp );

        for (int i = 0; i < nVoxels; i++)
        {
            uint32_t i0 = i * 8;
            uint32_t i1 = i * 8 + 1;
            uint32_t i2 = i * 8 + 2;
            uint32_t i3 = i * 8 + 3;
            uint32_t i4 = i * 8 + 4;
            uint32_t i5 = i * 8 + 5;
            uint32_t i6 = i * 8 + 6;
            uint32_t i7 = i * 8 + 7;

            uint8_t nverts = 4;
#define F( a, b, c, d ) fwrite(&nverts, sizeof(uint8_t), 1, fp); fwrite(&(a), sizeof(uint32_t), 1, fp ); fwrite(&(b), sizeof(uint32_t), 1, fp ); fwrite(&(c), sizeof(uint32_t), 1, fp ); fwrite(&(d), sizeof(uint32_t), 1, fp );
            // Left Hand
            F( i3, i2, i1, i0 );
            F( i4, i5, i6, i7 );
            F( i0, i1, i5, i4 );
            F( i1, i2, i6, i5 );
            F( i2, i3, i7, i6 );
            F( i3, i0, i4, i7 );
#undef F
        }

        fclose(fp);
    }
    std::vector<glm::vec3> points;
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

struct OctreeTask
{
    uint64_t morton;
    uint32_t child;
};
struct OctreeNode
{
    uint8_t mask;
    uint32_t children[8];
};
float maxElement(float a, float b, float c)
{
    return glm::max(glm::max(a, b), c);
}
float minElement(float a, float b, float c)
{
    return glm::min(glm::min(a, b), c);
}

glm::vec3 g_ro;
glm::vec3 g_rd;

void octreeTraverse(
    const std::vector<OctreeNode>& nodes, uint32_t nodeIndex,
    float tx0, float ty0, float tz0,
    float tx1, float ty1, float tz1, float *t0, float *t1, int depth = 0 )
{
    //if (g_rd.x < 0.0f)
    //{
    //    std::swap(tx0, tx1);
    //}
    //if (g_rd.y < 0.0f)
    //{
    //    std::swap(ty0, ty1);
    //}
    //if (g_rd.z < 0.0f)
    //{
    //    std::swap(tz0, tz1);
    //}

    float tmin = maxElement(tx0, ty0, tz0);
    float tmax = minElement(tx1, ty1, tz1);
    if (depth == 1)
    {
        pr::DrawSphere(g_ro + g_rd * tmin, 0.05f, { 0 ,255,255 });
        pr::DrawSphere(g_ro + g_rd * tmax, 0.05f, { 0 ,0,255 });
    }

    if (!(tmin < tmax))
    {
        return;
    }
    if( nodeIndex == -1 )
    {
        *t0 = glm::min( *t0, tmin );
        *t1 = glm::max( *t1, tmax );
        return;
    }

    float txM = 0.5f * (tx0 + tx1);
    float tyM = 0.5f * (ty0 + ty1);
    float tzM = 0.5f * (tz0 + tz1);

    const OctreeNode& node = nodes[nodeIndex];
    if( node.mask & ( 0x1 ))
        octreeTraverse(nodes, node.children[0], tx0, ty0, tz0, txM, tyM, tzM, t0, t1 );

    if (node.mask & (0x1 << 1 ))
        octreeTraverse(nodes, node.children[1], txM, ty0, tz0, tx1, tyM, tzM, t0, t1);

    if (node.mask & (0x1 << 2))
        octreeTraverse(nodes, node.children[2], tx0, tyM, tz0, txM, ty1, tzM, t0, t1);

    if (node.mask & (0x1 << 3))
        octreeTraverse(nodes, node.children[3], txM, tyM, tz0, tx1, ty1, tzM, t0, t1);

    if (node.mask & (0x1 << 4))
        octreeTraverse(nodes, node.children[4], tx0, ty0, tzM, txM, tyM, tz1, t0, t1);

    if (node.mask & (0x1 << 5))
        octreeTraverse(nodes, node.children[5], txM, ty0, tzM, tx1, tyM, tz1, t0, t1, depth + 1);

    if (node.mask & (0x1 << 6))
        octreeTraverse(nodes, node.children[6], tx0, tyM, tzM, txM, ty1, tz1, t0, t1);
        
    if (node.mask & (0x1 << 7))
        octreeTraverse(nodes, node.children[7], txM, txM, tzM, tx1, ty1, tz1, t0, t1);
}

void drawAABBscaled( glm::vec3 lower, glm::vec3 upper, float scale, glm::u8vec3 color, float lineWidth = 1.0f )
{
    glm::vec3 c = ( lower + upper ) * 0.5f;
    glm::vec3 h = ( upper - lower ) * 0.5f;
    pr::DrawAABB( c - h * scale, c + h * scale, color, lineWidth );
}

void octreeTraverseNaive(
    const std::vector<OctreeNode>& nodes, uint32_t nodeIndex,
    glm::vec3 ro,
    glm::vec3 one_over_rd,
    glm::vec3 lower,
    glm::vec3 upper, 
    float* t, int* nMajor, int depth )
{
    glm::vec3 t0v = (lower - ro) * one_over_rd;
    glm::vec3 t1v = (upper - ro) * one_over_rd;
    glm::vec3 tmin = glm::min(t0v, t1v);
    glm::vec3 tmax = glm::max(t0v, t1v);
    float a = maxElement( tmin.x, tmin.y, tmin.z );
    float b = minElement( tmax.x, tmax.y, tmax.z );

    // TODO 0 <= t
    if( b < a )
    {
        return;
    }
    if( nodeIndex == -1 )
    {
        if (a < *t)
        {
            *t = a;

            if (a == tmin.x)
            {
                *nMajor = 1;
            }
            else if (a == tmin.y)
            {
                *nMajor = 2;
            }
            else
            {
                *nMajor = 0;
            }
        }
        return;
    }
    const OctreeNode& node = nodes[nodeIndex];
    glm::vec3 mid = (lower + upper) * 0.5f;
    
    for( uint32_t i = 0; i < 8; i++ )
    {
        glm::vec3 l;
        glm::vec3 u;
        l.x = (i & 0x1) ? mid.x : lower.x;
        u.x = (i & 0x1) ? upper.x : mid.x;
        l.y = (i & 0x2) ? mid.y : lower.y;
        u.y = (i & 0x2) ? upper.y : mid.y;
        l.z = (i & 0x4) ? mid.z : lower.z;
        u.z = (i & 0x4) ? upper.z : mid.z;

        if (node.mask & ( 0x1u << i ) )
        {
            octreeTraverseNaive(nodes, node.children[i], ro, one_over_rd,
                l,
                u,
                t, nMajor, depth + 1
            );
        }
    }
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

    const char* input = "bunny.obj";
    // const char* input = "Tri.obj";
    SetDataDir(ExecutableDir());
    std::string errorMsg;
    std::shared_ptr<FScene> scene = ReadWavefrontObj( GetDataPath(input), errorMsg );

    while (pr::NextFrame() == false) {
        if (IsImGuiUsingMouse() == false) {
            UpdateCameraBlenderLike(&camera);
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
        static int gridRes = 16;

        static glm::vec3 from = { 5 , 5, 5 };
        static glm::vec3 to = { 0,  0, 0.0f };

        ManipulatePosition(camera, &from, 1);
        ManipulatePosition(camera, &to, 1);

        DrawText(from, "from");
        DrawText(to, "to");
        DrawLine(from, to, { 128 , 128 , 128 });


        std::vector<OctreeNode> nodes;
        glm::vec3 octree_lower;
        glm::vec3 octree_upper;

        scene->visitPolyMesh([&nodes, &octree_lower, &octree_upper](std::shared_ptr<const FPolyMeshEntity> polymesh) {
            ColumnView<int32_t> faceCounts(polymesh->faceCounts());
            ColumnView<int32_t> indices(polymesh->faceIndices());
            ColumnView<glm::vec3> positions(polymesh->positions());

            // Geometry
            pr::PrimBegin(pr::PrimitiveMode::Lines);
            for (int i = 0; i < positions.count(); i++)
            {
                glm::vec3 p = positions[i];
                glm::ivec3 color = { 255,255,255 };
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

            glm::vec3 lower = glm::vec3(FLT_MAX, FLT_MAX, FLT_MAX);
            glm::vec3 upper = glm::vec3(-FLT_MAX, -FLT_MAX, -FLT_MAX);

            for (int i = 0; i < faceCounts.count(); i++)
            {
                for (int j = 0; j < 3; ++j)
                {
                    int index = indices[i * 3 + j];
                    lower = glm::min(lower, positions[index]);
                    upper = glm::max(upper, positions[index]);
                }
            }

            // bounding box
            glm::vec3 size = upper - lower;
            float dps = glm::max(glm::max(size.x, size.y), size.z) / (float)gridRes;

            octree_lower = lower;
            octree_upper = lower + glm::vec3(dps, dps, dps) * (float)gridRes;

            DrawAABB(lower, lower + glm::vec3(dps, dps, dps) * (float)gridRes, { 255 ,0 ,0 });

            // Voxelization
            static std::vector<char> voxels;
            voxels.resize(gridRes * gridRes * gridRes);
            std::fill(voxels.begin(), voxels.end(), 0);

            std::set<uint64_t> mortonVoxels;

            Stopwatch voxelsw;

            glm::vec3 origin = lower;

            for (int i = 0; i < faceCounts.count(); i++)
            {
                glm::vec3 v0 = positions[indices[i * 3]];
                glm::vec3 v1 = positions[indices[i * 3 + 1]];
                glm::vec3 v2 = positions[indices[i * 3 + 2]];

                VTContext context(v0, v1, v2, sixSeparating, origin, dps, gridRes);
                glm::ivec2 xrange = context.xRangeInclusive();
                for (int x = xrange.x; x <= xrange.y; x++)
                {
                    glm::ivec2 yrange = context.yRangeInclusive(x, dps);
                    for (int y = yrange.x; y <= yrange.y; y++)
                    {
                        glm::ivec2 zrange = context.zRangeInclusive(x, y, dps, sixSeparating);
                        for (int z = zrange.x; z <= zrange.y; z++)
                        {
                            glm::vec3 p = context.p(x, y, z, dps);
                            if (context.intersect(p))
                            {
                                glm::ivec3 c = context.i(x, y, z);
                                SequencialHasher h;
                                h.add(c.x, gridRes);
                                h.add(c.y, gridRes);
                                h.add(c.z, gridRes);
                                voxels[h.value()] = 1;

                                mortonVoxels.insert(encode2mortonCode_PDEP(c.x, c.y, c.z));
                            }
                        }
                    }
                }
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
                    // DrawAABB(p, p + glm::vec3(dps, dps, dps), { 200 ,200 ,200 });

                    drawAABBscaled(p, p + glm::vec3(dps, dps, dps), 0.97f, { 200 ,200 ,200 });
                }
            }

            uint32_t wide = gridRes;


            std::vector<OctreeTask> curTasks;
            for (auto m : mortonVoxels)
            {
                OctreeTask task;
                task.morton = m;
                task.child = -1;
                curTasks.push_back( task );
            }
            std::vector<OctreeTask> nextTasks;

            while( 1 < wide )
            {
                std::vector<OctreeTask> sameParent;
                uint64_t parent = -1;

                auto emit = [&]()
                {
                    // allocate
                    uint32_t c = nodes.size(); 
                    nodes.push_back(OctreeNode());
                    nodes[c].mask = 0;
                    for (int i = 0; i < 8; i++)
                    {
                        nodes[c].children[i] = -1;
                    }

                    // set child
                    PR_ASSERT(sameParent.size() <= 8);
                    for (int i = 0; i < sameParent.size(); i++)
                    {
                        uint32_t space = sameParent[i].morton & 0x7;
                        nodes[c].mask |= (1 << space) & 0xFF;
                        nodes[c].children[space] = sameParent[i].child;
                    }

                    OctreeTask nextTask;
                    nextTask.morton = parent;
                    nextTask.child = c;
                    nextTasks.push_back( nextTask );

                    parent = -1;
                    sameParent.clear();
                };

                for (int i = 0; i < curTasks.size(); i++)
                {
                    if( parent == -1 )
                    {
                        sameParent.push_back(curTasks[i]);
                        parent = curTasks[i].morton >> 3;
                        continue;
                    }

                    if (parent == (curTasks[i].morton >> 3))
                    {
                        sameParent.push_back(curTasks[i]);
                    }
                    else
                    {
                        emit();
                        sameParent.push_back(curTasks[i]);
                        parent = curTasks[i].morton >> 3;
                    }
                }

                if (sameParent.size())
                {
                    emit();
                }

                curTasks.clear();
                std::swap(curTasks, nextTasks);

                wide /= 2;
            }
        });

        glm::vec3 ro = from;
        glm::vec3 rd = to - from;
        glm::vec3 one_over_rd = glm::vec3(1.0f) / rd;

        glm::vec3 t0 = (octree_lower - ro) * one_over_rd;
        glm::vec3 t1 = (octree_upper - ro) * one_over_rd;
        //glm::vec3 tmin = glm::min(t0, t1);
        //glm::vec3 tmax = glm::max(t0, t1);
        glm::vec3 tmin = t0;
        glm::vec3 tmax = t1;

        float a = maxElement(tmin.x, tmin.y, tmin.z);
        float b = minElement(tmax.x, tmax.y, tmax.z);

        //DrawSphere(ro + rd * a, 0.05f, { 255,0,0 });
        //DrawSphere(ro + rd * b, 0.05f, { 255,0,0 });

        g_ro = ro;
        g_rd = rd;

        //float rt0 = FLT_MAX;
        //float rt1 = 0;
        // octreeTraverse( nodes, nodes.size() - 1,  tmin.x, tmin.y, tmin.z, tmax.x, tmax.y, tmax.z, &rt0, &rt1 );

        float rt0 = FLT_MAX;
        int nMajor;
        octreeTraverseNaive( nodes, nodes.size() - 1, ro, one_over_rd, octree_lower, octree_upper, &rt0, &nMajor, 0 );

        DrawSphere(ro + rd * rt0, 0.01f, { 255,0,0 });

        glm::vec3 hitN = unProjectPlane( { 0.0f, 0.0f }, project2plane_reminder( rd, nMajor ) < 0.0f ? 1.0f : -1.0f , nMajor );
        DrawArrow(ro + rd * rt0, ro + rd * rt0 + hitN * 0.1f, 0.01f, { 255,0,0 });
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
        ManipulatePosition(camera, &origin, 1);
        DrawAABB(origin, origin + glm::vec3(dps, dps, dps) * (float)gridRes, { 255 ,0 ,0 });

        // VoxelTriangleIntersector intersector( v0, v1, v2, sixSeparating, dps );

        VTContext context(v0, v1, v2, sixSeparating, origin, dps, gridRes);
        glm::ivec2 xrange = context.xRangeInclusive();
        for (int x = xrange.x; x <= xrange.y; x++)
        {
            glm::ivec2 yrange = context.yRangeInclusive(x, dps);
            for (int y = yrange.x; y <= yrange.y; y++)
            {
                glm::ivec2 zrange = context.zRangeInclusive(x, y, dps, sixSeparating);
                for (int z = zrange.x; z <= zrange.y; z++)
                {
                    glm::vec3 p = context.p(x, y, z, dps);
                    if (context.intersect(p))
                    {
                        DrawAABB(p, p + glm::vec3(dps, dps, dps), { 200 ,200 ,200 });
                    }
                }
            }
        }
#endif
        

        //{
        //    Xoshiro128StarStar random;
        //    for (int i = 0; i < 1000000000; i++)
        //    {
        //        uint32_t x = random.uniformi() & 0x1FFFFF; // 21 bits
        //        uint32_t y = random.uniformi() & 0x1FFFFF; // 21 bits
        //        uint32_t z = random.uniformi() & 0x1FFFFF; // 21 bits

        //        uint64_t m0 = encode2mortonCode(x, y, z);
        //        uint64_t m1 = encode2mortonCode_PDEP(x, y, z);
        //        uint64_t m2 = encode2mortonCode_magicbits(x, y, z);
        //        PR_ASSERT(m0 == m1);
        //        PR_ASSERT(m0 == m2);
        //    }
        //}

        // perf
        //{
        //    uint64_t k = 0;
        //    Stopwatch sw;
        //    Xoshiro128StarStar random;
        //    for (int i = 0; i < 100000000; i++)
        //    {
        //        uint32_t x = random.uniformi() & 0x1FFFFF; // 21 bits
        //        uint32_t y = random.uniformi() & 0x1FFFFF; // 21 bits
        //        uint32_t z = random.uniformi() & 0x1FFFFF; // 21 bits

        //        uint64_t m0 = encode2mortonCode(x, y, z);
        //        k += m0;
        //    }
        //    printf("%f s encode2mortonCode, %lld\n", sw.elapsed(), k);
        //}
        //{
        //    uint64_t k = 0;
        //    Stopwatch sw;
        //    Xoshiro128StarStar random;
        //    for (int i = 0; i < 100000000; i++)
        //    {
        //        uint32_t x = random.uniformi() & 0x1FFFFF; // 21 bits
        //        uint32_t y = random.uniformi() & 0x1FFFFF; // 21 bits
        //        uint32_t z = random.uniformi() & 0x1FFFFF; // 21 bits

        //        uint64_t m0 = encode2mortonCode_magicbits(x, y, z);
        //        k += m0;
        //    }
        //    printf("%f s encode2mortonCode_magicbits, %lld\n", sw.elapsed(), k);
        //}
        //{
        //    uint64_t k = 0;
        //    Stopwatch sw;
        //    Xoshiro128StarStar random;
        //    for (int i = 0; i < 100000000; i++)
        //    {
        //        uint32_t x = random.uniformi() & 0x1FFFFF; // 21 bits
        //        uint32_t y = random.uniformi() & 0x1FFFFF; // 21 bits
        //        uint32_t z = random.uniformi() & 0x1FFFFF; // 21 bits

        //        uint64_t m0 = encode2mortonCode_PDEP(x, y, z);
        //        k += m0;
        //    }
        //    printf("%f s encode2mortonCode_PDEP, %lld\n", sw.elapsed(), k);
        //}

        // morton code
#if 0
        static int voxelX = 0;
        static int voxelY = 0;
        static int voxelZ = 0;

        glm::vec3 p = { voxelX, voxelY, voxelZ };
        DrawAABB( p + glm::vec3(0.01f), p + glm::vec3(1.0f, 1.0f, 1.0f) - glm::vec3(0.01f), {255, 255 ,0});

        uint64_t code = encode2mortonCode( voxelX, voxelY, voxelZ );

        static int NDepth = 2;

        float box = (float)( 1u << NDepth );
        glm::vec3 lower = { 0.0f, 0.0f, 0.0f };
        glm::vec3 upper = { box, box, box };

        DrawAABB(lower, upper, { 255,255,255 });
        
        for ( int i = 0; i < NDepth; i++ )
        {
            int s = ( NDepth - i - 1) * 3;
            uint64_t space = ( code & ( 7LLU << s ) ) >> s;

            box *= 0.5f;

            if (space & 0x01)
            {
                lower.x += box;
            }
            else
            {
                upper.x -= box;
            }

            if( space & 0x02 )
            {
                lower.y += box;
            }
            else
            {
                upper.y -= box;
            }

            if (space & 0x04)
            {
                lower.z += box;
            }
            else
            {
                upper.z -= box;
            }
            DrawAABB(lower, upper, { 255,255,255 });
        }
#endif

        PopGraphicState();
        EndCamera();

        BeginImGui();

        ImGui::SetNextWindowSize({ 500, 800 }, ImGuiCond_Once);
        ImGui::Begin("Panel");
        ImGui::Text("fps = %f", GetFrameRate());


        //ImGui::InputInt("voxelX", &voxelX);
        //ImGui::InputInt("voxelY", &voxelY);
        //ImGui::InputInt("voxelZ", &voxelZ);

        //ImGui::Text("%llu", encode2mortonCode(voxelX, voxelY, voxelZ));
        //
        //ImGui::InputInt("NDepth", &NDepth);
        
#if 1
        ImGui::InputFloat("dps", &dps, 0.01f);
        ImGui::InputInt("gridRes", &gridRes);
        ImGui::Checkbox("sixSeparating", &sixSeparating);
        ImGui::Checkbox("experiment", &experiment);
        ImGui::Text("voxel: %f s", voxel_time);

        static int Resolution = 1024;
        ImGui::InputInt("Resolution", &Resolution);
        if (ImGui::Button("Voxelize As Mesh"))
        {
            const char* input = "bunny.obj";
            // const char* input = "Tri.obj";
            std::string errorMsg;
            std::shared_ptr<FScene> scene = ReadWavefrontObj(GetDataPath(input), errorMsg);
            std::shared_ptr<FPolyMeshEntity> polymesh = std::dynamic_pointer_cast<FPolyMeshEntity>(scene->entityAt(0));
            ColumnView<int32_t> faceCounts(polymesh->faceCounts());
            ColumnView<int32_t> indices(polymesh->faceIndices());
            ColumnView<glm::vec3> positions(polymesh->positions());

            // Assume Triangle
            glm::vec3 lower = glm::vec3(FLT_MAX, FLT_MAX, FLT_MAX);
            glm::vec3 upper = glm::vec3(-FLT_MAX, -FLT_MAX, -FLT_MAX);

            for (int i = 0; i < faceCounts.count(); i++)
            {
                for (int j = 0; j < 3; ++j)
                {
                    int index = indices[i * 3 + j];
                    lower = glm::min(lower, positions[index]);
                    upper = glm::max(upper, positions[index]);
                }
            }

            // bounding box
            glm::vec3 size = upper - lower;

            float dps = glm::max(glm::max(size.x, size.y), size.z) / (float)Resolution;

            // Voxelization
            std::set<uint32_t> voxels;
            VoxelObjWriter writer;
            // Stopwatch voxelsw;

            glm::vec3 origin = lower;

            for (int i = 0; i < faceCounts.count(); i++)
            {
                glm::vec3 v0 = positions[indices[i * 3]];
                glm::vec3 v1 = positions[indices[i * 3 + 1]];
                glm::vec3 v2 = positions[indices[i * 3 + 2]];

                VTContext context(v0, v1, v2, sixSeparating, origin, dps, Resolution);
                glm::ivec2 xrange = context.xRangeInclusive();
                for (int x = xrange.x; x <= xrange.y; x++)
                {
                    glm::ivec2 yrange = context.yRangeInclusive(x, dps);
                    for (int y = yrange.x; y <= yrange.y; y++)
                    {
                        glm::ivec2 zrange = context.zRangeInclusive(x, y, dps, sixSeparating);
                        for (int z = zrange.x; z <= zrange.y; z++)
                        {
                            glm::vec3 p = context.p(x, y, z, dps);
                            if (context.intersect(p))
                            {
                                glm::ivec3 c = context.i(x, y, z);
                                SequencialHasher h;
                                h.add(c.x, Resolution);
                                h.add(c.y, Resolution);
                                h.add(c.z, Resolution);

                                if (voxels.count(h.value()) == 0 )
                                {
                                    writer.add(p, dps);
                                }
                                else
                                {
                                    voxels.insert(h.value());
                                }
                            }
                        }
                    }
                }
            }

            writer.savePLY(GetDataPath("vox.ply").c_str());
        }
#endif
        
        ImGui::End();

        EndImGui();
    }

    pr::CleanUp();
}
