#pragma once
#include <inttypes.h>
#include <string>
#include <vector>
#include "Orochi/Orochi.h"
#include <intrin.h>

inline uint64_t div_round_up64( uint64_t val, uint64_t divisor )
{
	return ( val + divisor - 1 ) / divisor;
}
inline uint64_t next_multiple64( uint64_t val, uint64_t divisor )
{
	return div_round_up64( val, divisor ) * divisor;
}


#define HIPUTIL_ASSERT( ExpectTrue ) \
	if( ( ExpectTrue ) == 0 )     \
	{                             \
		__debugbreak();           \
	}

inline void loadFileAsVector( std::vector<char>* buffer, const char* fllePath )
{
    FILE* fp = fopen( fllePath, "rb" );
    if( fp == nullptr )
    {
        return;
    }

    fseek( fp, 0, SEEK_END );

    buffer->resize( ftell( fp ) );

    fseek( fp, 0, SEEK_SET );

    size_t s = fread( buffer->data(), 1, buffer->size(), fp );
    if( s != buffer->size() )
    {
        buffer->clear();
        return;
    }
    fclose( fp );
    fp = nullptr;
}

class Buffer
{
public:
    Buffer( const Buffer& ) = delete;
    void operator=( const Buffer& ) = delete;

    Buffer( int64_t bytes )
        : m_bytes( std::max( bytes, 1LL ) )
    {
        oroMalloc( &m_ptr, m_bytes );
    }
    ~Buffer()
    {
        oroFree( m_ptr );
    }
    int64_t bytes() const
    {
        return m_bytes;
    }
    char* data()
    {
        return (char*)m_ptr;
    }
private:
    int64_t m_bytes;
    oroDeviceptr m_ptr;
};

struct ShaderArgument
{
    template <class T>
    void add( T p )
    {
        int bytes = sizeof( p );
        int location = m_buffer.size();
        m_buffer.resize( m_buffer.size() + bytes );
        memcpy( m_buffer.data() + location, &p, bytes );
        m_locations.push_back( location );
    }
    void clear()
    {
        m_buffer.clear();
        m_locations.clear();
    }

    std::vector<void*> kernelParams() const
    {
        std::vector<void*> ps;
        for( int i = 0; i < m_locations.size(); i++ )
        {
            ps.push_back( (void*)( m_buffer.data() + m_locations[i] ) );
        }
        return ps;
    }

private:
    std::vector<char> m_buffer;
    std::vector<int> m_locations;
};
class Shader
{
public:
    Shader( const char* src, const char* kernelLabel, const std::vector<std::string>& extraArgs )
    {
        orortcProgram program = 0;
        orortcCreateProgram( &program, src, kernelLabel, 0, 0, 0 );
        std::vector<std::string> options;

        for( int i = 0; i < extraArgs.size(); ++i )
        {
            options.push_back( extraArgs[i] );
        }

        std::vector<const char*> optionChars;
        for( int i = 0; i < options.size(); ++i )
        {
            optionChars.push_back( options[i].c_str() );
        }

        orortcResult compileResult = orortcCompileProgram( program, optionChars.size(), optionChars.data() );

        size_t logSize = 0;
        orortcGetProgramLogSize( program, &logSize );
        if( 1 < logSize )
        {
            std::vector<char> compileLog( logSize );
            orortcGetProgramLog( program, compileLog.data() );
            printf( "%s", compileLog.data() );
        }
        
        HIPUTIL_ASSERT( compileResult == ORORTC_SUCCESS );

        size_t codeSize = 0;
        orortcGetCodeSize( program, &codeSize );

        m_shaderBinary.resize( codeSize );
        orortcGetCode( program, m_shaderBinary.data() );

        // FILE* fp = fopen( "shader.bin", "wb" );
        // fwrite( m_shaderBinary.data(), m_shaderBinary.size(), 1, fp );
        // fclose( fp );

        orortcDestroyProgram( &program );

        oroError e = oroModuleLoadData( &m_module, m_shaderBinary.data() );
		HIPUTIL_ASSERT( e == oroSuccess );
    }
    ~Shader()
    {
        oroModuleUnload( m_module );
    }
    void launch( const char* name,
                    const ShaderArgument& arguments,
                    unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                    unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
                    oroStream hStream )
    {
        if( m_functions.count( name ) == 0 )
        {
            oroFunction f = 0;
            oroError e = oroModuleGetFunction( &f, m_module, name );
			HIPUTIL_ASSERT( e == oroSuccess );
            m_functions[name] = f;
        }

        auto params = arguments.kernelParams();
        oroFunction f = m_functions[name];
        oroError e = oroModuleLaunchKernel( f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, 0, hStream, params.data(), 0 );
		HIPUTIL_ASSERT( e == oroSuccess );
    }

private:
    oroModule m_module = 0;
    std::map<std::string, oroFunction> m_functions;
    std::vector<char> m_shaderBinary;
};