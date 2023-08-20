#pragma once

#include <stdint.h>
#include <glm/glm.hpp>
#include <set>

#include "morton.hpp"
#include "voxCommon.hpp"


inline void buildOctreeDAGReference( std::vector<OctreeNode>* nodes, const std::vector<uint64_t>& mortonVoxels, int wide )
{
	nodes->clear();

	std::vector<OctreeTask> curTasks;
	for( auto m : mortonVoxels )
	{
		OctreeTask task;
		task.morton = m;
		task.child = -1;
		task.numberOfVoxels = 1;
		curTasks.push_back( task );
	}
	std::vector<OctreeTask> nextTasks;

	struct MortonGroup
	{
		int beg;
		int end;
	};

	// Reference
	std::map<OctreeNode, int> existings;

	while( 1 < wide )
	{
		// make groups
		std::vector<MortonGroup> groups;
		MortonGroup group = { -1, -1 };
		for( int i = 0; i < curTasks.size(); i++ )
		{
			if( group.beg == -1 )
			{
				group.beg = i;
				group.end = i + 1;
				// parent = curTasks[i].morton >> 3;
				continue;
			}

			uint64_t pMorton = curTasks[group.beg].getMortonParent(); // Parent
			if( pMorton == ( curTasks[i].getMortonParent() ) )
			{
				group.end = i + 1;
			}
			else
			{
				groups.push_back( group );
				group.beg = i;
				group.end = i + 1;
			}
		}
		if( group.beg != -1 )
		{
			groups.push_back( group );
		}

		for( int i = 0; i < groups.size(); i++ )
		{
			MortonGroup group = groups[i];

			OctreeNode node;
			node.mask = 0;
			for( int j = 0; j < 8; j++ )
			{
				node.children[j] = -1;
				node.nVoxelsPSum[j] = 0;
			}

			// set child
			for( int j = group.beg; j < group.end; j++ )
			{
				uint32_t space = curTasks[j].morton & 0x7;
				node.mask |= ( 1 << space ) & 0xFF;
				node.children[space] = curTasks[j].child;
				node.nVoxelsPSum[space] = curTasks[j].numberOfVoxels;
			}

			// prefix scan exclusive
			int numberOfVoxels = 0;
			for( int j = 0; j < 8; j++ )
			{
				uint32_t c = node.nVoxelsPSum[j];
				node.nVoxelsPSum[j] = numberOfVoxels;
				numberOfVoxels += c;
			}

			uint32_t nodeIndex;

			auto it = existings.find( node );
			if( it == existings.end() )
			{
				nodeIndex = nodes->size();
				nodes->push_back( node );
				existings[node] = nodeIndex;
			}
			else
			{
				nodeIndex = it->second;
			}

			OctreeTask nextTask;
			nextTask.morton = curTasks[group.beg].getMortonParent();
			nextTask.child = nodeIndex;
			nextTask.numberOfVoxels = numberOfVoxels;
			nextTasks.push_back( nextTask );
		}

		curTasks.clear();
		std::swap( curTasks, nextTasks );

		wide /= 2;
	}
}

inline void buildOctreeNaive( std::vector<OctreeNode>* nodes, const std::vector<uint64_t>& mortonVoxels, int wide )
{
	nodes->clear();

	std::vector<OctreeTask> curTasks;
	for( auto m : mortonVoxels )
	{
		OctreeTask task;
		task.morton = m;
		task.child = -1;
		curTasks.push_back( task );
	}
	std::vector<OctreeTask> nextTasks;

	struct MortonGroup
	{
		int beg;
		int end;
	};

	while( 1 < wide )
	{
		// make groups
		std::vector<MortonGroup> groups;
		MortonGroup group = { -1, -1 };
		for( int i = 0; i < curTasks.size(); i++ )
		{
			if( group.beg == -1 )
			{
				group.beg = i;
				group.end = i + 1;
				// parent = curTasks[i].morton >> 3;
				continue;
			}

			uint64_t pMorton = curTasks[group.beg].morton >> 3; // Parent
			if( pMorton == ( curTasks[i].morton >> 3 ) )
			{
				group.end = i + 1;
			}
			else
			{
				groups.push_back( group );
				group.beg = i;
				group.end = i + 1;
			}
		}
		if( group.beg != -1 )
		{
			groups.push_back( group );
		}

		// build nodes
		for( int i = 0; i < groups.size(); i++ )
		{
			MortonGroup group = groups[i];

			OctreeNode node;
			node.mask = 0;
			for( int j = 0; j < 8; j++ )
			{
				node.children[j] = -1;
			}

			// set child
			for( int j = group.beg; j < group.end; j++ )
			{
				uint32_t space = curTasks[j].morton & 0x7;
				node.mask |= ( 1 << space ) & 0xFF;
				node.children[space] = curTasks[j].child;
				node.nVoxelsPSum[space] = 0; // not supported yet.
			}

			uint32_t c = nodes->size();
			nodes->push_back( node );

			OctreeTask nextTask;
			nextTask.morton = curTasks[group.beg].morton >> 3;
			nextTask.child = c;
			nextTasks.push_back( nextTask );
		}

		curTasks.clear();
		std::swap( curTasks, nextTasks );

		wide /= 2;
	}
}

class IntersectorOctree
{
public:
	IntersectorOctree()
	{
	}

	void build( const std::vector<uint64_t>& mortonVoxels, const glm::vec3& origin, float dps, int gridRes )
	{
		m_lower = origin;
		m_upper = origin + glm::vec3( dps, dps, dps ) * (float)gridRes;
		buildOctreeNaive( &m_nodes, mortonVoxels, gridRes );
		embedMasks();
	}

	void buildDAGReference( const std::vector<uint64_t>& mortonVoxels, const glm::vec3& origin, float dps, int gridRes )
	{
		m_lower = origin;
		m_upper = origin + glm::vec3( dps, dps, dps ) * (float)gridRes;
		buildOctreeDAGReference( &m_nodes, mortonVoxels, gridRes );
		embedMasks();
	}

	void embedMasks()
	{
#if defined( ENABLE_EMBEDED_MASK )
		assert( m_nodes.size() < 0xFFFFFF );
		for( int i = 0; i < m_nodes.size(); i++ )
		{
			embedMask( m_nodes.data(), i );
		}
#endif
	}

	void intersect( const float3& ro, const float3& rd, float* t, int* nMajor, uint32_t* vIndex )
	{
		StackElement stack[32];
		octreeTraverse_EfficientParametric( 
			m_nodes.data(), m_nodes.size() - 1, 
			stack,
			ro, rd, 
			{ m_lower.x, m_lower.y, m_lower.z },
			{ m_upper.x, m_upper.y, m_upper.z }, t, nMajor, vIndex, false /* isShadowRay */ );
	}

	uint64_t getMemoryConsumption()
	{
		return m_nodes.size() * sizeof( OctreeNode );
	}
	glm::vec3 m_lower = glm::vec3( 0.0f );
	glm::vec3 m_upper = glm::vec3( 0.0f );
	std::vector<OctreeNode> m_nodes;
};