#pragma once
#include "vectorMath.hpp"

#if !defined( __CUDACC__ ) && !defined( __HIPCC__ )
#include "Orochi/Orochi.h"
#include "Orochi/OrochiUtils.h"
#endif

template <int N>
class BitSet
{
public:
	enum
	{
		length = div_round_up( N, 32 )
	};
	DEVICE BitSet()
	{
		for( int i = 0; i < length; i++ )
		{
			m_data[i] = 0;
		}
	}
	DEVICE void set( int i, bool value )
	{
		int index = i / 32;
		int bit = i % 32;
		if( value )
		{
			m_data[index] |= ( 1u << bit );
		}
		else
		{
			m_data[index] &= ~( 1u << bit );
		}
	}
	DEVICE bool get( int i ) const
	{
		int index = i / 32;
		int bit = i % 32;
		return m_data[index] & ( 1u << bit );
	}
	uint32_t m_data[length];
};

struct StreamCompaction
{
#if !defined( __CUDACC__ ) && !defined( __HIPCC__ )
	StreamCompaction()
	{
		oroMalloc( (oroDeviceptr*)&m_iterator, sizeof( uint64_t ) );
		oroMalloc( (oroDeviceptr*)&m_blockCounter, sizeof( uint32_t ) );
	}
	~StreamCompaction()
	{
		oroFree( (oroDeviceptr)m_iterator );
		oroFree( (oroDeviceptr)m_blockCounter );
	}
	StreamCompaction( const StreamCompaction& ) = delete;
	void operator=( const StreamCompaction& ) = delete;

	void clear( oroStream stream )
	{
		oroMemsetD32Async( (oroDeviceptr)m_iterator, 0, 2, stream );
		oroMemsetD32Async( (oroDeviceptr)m_blockCounter, 0, 1, stream );
	}
	uint32_t readCounter( oroStream stream )
	{
		uint32_t n = 0;
		oroMemcpyDtoHAsync( &n, (oroDeviceptr)m_iterator, sizeof( uint32_t ), stream );
		oroStreamSynchronize( stream );
		return n;
	}
#else
	enum
	{
		VOTE_BITS = 32
	};

	// Predicate ( srcIndex ) => bool
	//     srcIndex: an index of the source item
	//     return  : true if the item is needed.
	// Store ( srcIndex, dstIndex )
	//     srcIndex: an index of the source item
	//     dstindex: an index of the destination item
	//
	template <int ITEMS_PER_BLOCK, int BLOCK_DIM, class Predicate, class Store>
	DEVICE void filter( Predicate predicate, Store store, int* nOutputsInTheBlock, int* globalPrefixInTheBlock )
	{
		constexpr int nMasks = div_round_up( BLOCK_DIM, VOTE_BITS );
		constexpr int nItemPerThread = div_round_up( ITEMS_PER_BLOCK, BLOCK_DIM );

		__shared__ uint32_t gp;
		__shared__ uint32_t localSum;
		__shared__ uint32_t masks[nMasks];

		BitSet<nItemPerThread> votes;

		// clear
		if( threadIdx.x == 0 )
		{
			localSum = 0;
		}

		__syncthreads();

		for( int i = 0; i < ITEMS_PER_BLOCK; i += BLOCK_DIM )
		{
			int itemIndex = blockIdx.x * ITEMS_PER_BLOCK + i + threadIdx.x;
			if( i + threadIdx.x < ITEMS_PER_BLOCK && predicate( itemIndex ) )
			{
				votes.set( i / BLOCK_DIM, true );
				atomicInc( &localSum, 0xFFFFFFFF );
			}
		}

		__syncthreads();

		*nOutputsInTheBlock = localSum;

		if( threadIdx.x == 0 )
		{
			uint32_t sum = localSum;

			uint64_t expected;
			uint64_t cur = *m_iterator;
			uint32_t globalPrefix = cur & 0xFFFFFFFF;
			do
			{
				expected = (uint64_t)globalPrefix + ( (uint64_t)( blockIdx.x ) << 32 );
				uint64_t newValue = (uint64_t)globalPrefix + sum | ( (uint64_t)( blockIdx.x + 1 ) << 32 );
				cur = atomicCAS( m_iterator, expected, newValue );
				globalPrefix = cur & 0xFFFFFFFF;

			} while( cur != expected );

			gp = globalPrefix;
		}

		__syncthreads();

		uint32_t globalPrefix = gp;
		*globalPrefixInTheBlock = globalPrefix;

		for( int i = 0; i < ITEMS_PER_BLOCK; i += BLOCK_DIM )
		{
			// get current prefix
			uint32_t prefix = gp;

			// clear mask
			if( threadIdx.x < nMasks )
			{
				masks[threadIdx.x] = 0;
			}

			__syncthreads();

			int lane = threadIdx.x % VOTE_BITS;
			int index = threadIdx.x / VOTE_BITS;
			bool voted = votes.get( i / BLOCK_DIM );
			if( voted )
			{
				atomicOr( &masks[index], 1u << lane );
			}

			__syncthreads();

			if( voted )
			{
				uint32_t mask = masks[index];
				uint64_t lowerMask = ( 1u << lane ) - 1;
				uint32_t offset = __popc( mask & lowerMask );
				for( int j = 0; j < index; j++ )
				{
					offset += __popc( masks[j] );
				}
				int itemIndex = blockIdx.x * ITEMS_PER_BLOCK + i + threadIdx.x;
				store( itemIndex, prefix + offset, globalPrefix );
				atomicInc( &gp, 0xFFFFFFFF );
			}

			__syncthreads();
		}
	}

	DEVICE void granteeBlockExecutionOrder()
	{
		__threadfence();
		if( threadIdx.x == 0 )
		{
			while( atomicCAS( m_blockCounter, blockIdx.x, blockIdx.x + 1 ) != blockIdx.x )
				;
		}
		__threadfence();
		__syncthreads();
	}
#endif
	uint64_t* m_iterator;
	uint32_t* m_blockCounter;
};
