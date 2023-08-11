include "libs/PrLib"

workspace "Voxelization"
    location "build"
    configurations { "Debug", "Release" }
    startproject "main"

architecture "x86_64"

externalproject "prlib"
	location "libs/PrLib/build" 
    kind "StaticLib"
    language "C++"

function includePrLib()
    -- prlib
    -- setup command
    -- git submodule add https://github.com/Ushio/prlib libs/prlib
    -- premake5 vs2017
    dependson { "prlib" }
    includedirs { "libs/prlib/src" }
    libdirs { "libs/prlib/bin" }
    filter {"Debug"}
        links { "prlib_d" }
    filter {"Release"}
        links { "prlib" }
    filter{}
end

project "voxTriangle"
    kind "ConsoleApp"
    language "C++"
    targetdir "bin/"
    systemversion "latest"
    flags { "MultiProcessorCompile", "NoPCH" }

    -- Src
    files { "voxTriangle.cpp", "voxelization.hpp" }

    -- UTF8
    postbuildcommands { 
        "mt.exe -manifest ../utf8.manifest -outputresource:\"$(TargetDir)$(TargetName).exe\" -nologo"
    }

    includePrLib()

    symbols "On"

    filter {"Debug"}
        runtime "Debug"
        targetname ("voxTriangle_Debug")
        optimize "Off"
    filter {"Release"}
        runtime "Release"
        targetname ("voxTriangle")
        optimize "Full"
    filter{}

project "voxMesh"
    kind "ConsoleApp"
    language "C++"
    targetdir "bin/"
    systemversion "latest"
    flags { "MultiProcessorCompile", "NoPCH" }

    -- Src
    files { "voxMesh.cpp", "voxelization.hpp" }

    -- UTF8
    postbuildcommands { 
        "mt.exe -manifest ../utf8.manifest -outputresource:\"$(TargetDir)$(TargetName).exe\" -nologo"
    }

    includePrLib()

    symbols "On"

    filter {"Debug"}
        runtime "Debug"
        targetname ("voxMesh_Debug")
        optimize "Off"
    filter {"Release"}
        runtime "Release"
        targetname ("voxMesh")
        optimize "Full"
    filter{}
    
project "voxRT"
    kind "ConsoleApp"
    language "C++"
    targetdir "bin/"
    systemversion "latest"
    flags { "MultiProcessorCompile", "NoPCH" }

    -- Src
    files { "voxRT.cpp", "voxelization.hpp", "intersectorEmbree.hpp" }

    -- UTF8
    postbuildcommands { 
        "mt.exe -manifest ../utf8.manifest -outputresource:\"$(TargetDir)$(TargetName).exe\" -nologo"
    }

    includePrLib()

    -- Orochi ( include only )
    includedirs { "libs/orochi" }

    -- embree 4
    libdirs { "libs/prlib/libs/embree4/lib" }
    includedirs { "libs/prlib/libs/embree4/include" }
    links{
        "embree4",
        "tbb",
    }
    postbuildcommands {
        "{COPYFILE} ../libs/prlib/libs/embree4/bin/*.dll %{cfg.targetdir}/*.dll",
    }

    symbols "On"

    filter {"Debug"}
        runtime "Debug"
        targetname ("voxRT_Debug")
        optimize "Off"
    filter {"Release"}
        runtime "Release"
        targetname ("voxRT")
        optimize "Full"
    filter{}

project "voxRTGPU"
    kind "ConsoleApp"
    language "C++"
    targetdir "bin/"
    systemversion "latest"
    flags { "MultiProcessorCompile", "NoPCH" }

    -- Src
    files { "voxRTGPU.cpp", "voxelization.hpp", "hipUtil.hpp" }

    -- UTF8
    postbuildcommands { 
        "mt.exe -manifest ../utf8.manifest -outputresource:\"$(TargetDir)$(TargetName).exe\" -nologo"
    }

    -- Orochi
    includedirs { "libs/orochi" }
    files { "libs/orochi/Orochi/Orochi.h" }
    files { "libs/orochi/Orochi/Orochi.cpp" }
    includedirs { "libs/orochi/contrib/hipew/include" }
    files { "libs/orochi/contrib/hipew/src/hipew.cpp" }
    includedirs { "libs/orochi/contrib/cuew/include" }
    files { "libs/orochi/contrib/cuew/src/cuew.cpp" }
    links { "version" }
    postbuildcommands { 
        "{COPYFILE} ../libs/orochi/contrib/bin/win64/*.dll ../bin"
    }

    -- RadixSort
    includedirs { "libs/tinyhipradixsort" }

    includePrLib()

    symbols "On"

    filter {"Debug"}
        runtime "Debug"
        targetname ("voxRTGPU_Debug")
        optimize "Off"
    filter {"Release"}
        runtime "Release"
        targetname ("voxRTGPU")
        optimize "Full"
    filter{}

project "voxPTGPU"
    kind "ConsoleApp"
    language "C++"
    targetdir "bin/"
    systemversion "latest"
    flags { "MultiProcessorCompile", "NoPCH" }

    -- Src
    files { 
        "voxPTGPU.cpp", "voxUtil.hpp", "hipUtil.hpp", 
        "voxelization.hpp",  "IntersectorOctreeGPU.hpp", "voxCommon.hpp", "renderCommon.hpp", "pmjSampler" }

    -- UTF8
    postbuildcommands { 
        "mt.exe -manifest ../utf8.manifest -outputresource:\"$(TargetDir)$(TargetName).exe\" -nologo"
    }

    -- Orochi
    includedirs { "libs/orochi" }
    files { "libs/orochi/Orochi/Orochi.h" }
    files { "libs/orochi/Orochi/Orochi.cpp" }
    includedirs { "libs/orochi/contrib/hipew/include" }
    files { "libs/orochi/contrib/hipew/src/hipew.cpp" }
    includedirs { "libs/orochi/contrib/cuew/include" }
    files { "libs/orochi/contrib/cuew/src/cuew.cpp" }
    links { "version" }
    postbuildcommands { 
        "{COPYFILE} ../libs/orochi/contrib/bin/win64/*.dll ../bin"
    }

    -- RadixSort
    includedirs { "libs/tinyhipradixsort" }

    includePrLib()

    symbols "On"

    filter {"Debug"}
        runtime "Debug"
        targetname ("voxPTGPU_Debug")
        optimize "Off"
    filter {"Release"}
        runtime "Release"
        targetname ("voxPTGPU")
        optimize "Full"
    filter{}

project "unittest"
    kind "ConsoleApp"
    language "C++"
    targetdir "bin/"
    systemversion "latest"
    flags { "MultiProcessorCompile", "NoPCH" }

    -- Src
    files { "unittest.cpp", "voxelization.hpp" }

    -- Murmur3
    includedirs { "libs/smhasher" }
    files { "libs/smhasher/MurmurHash3.cpp", "libs/smhasher/MurmurHash3.cpp" }

    -- Orochi ( include only )
    includedirs { "libs/orochi" }
    
    -- UTF8
    postbuildcommands { 
        "mt.exe -manifest ../utf8.manifest -outputresource:\"$(TargetDir)$(TargetName).exe\" -nologo"
    }

    includePrLib()

    symbols "On"

    filter {"Debug"}
        runtime "Debug"
        targetname ("unittest_Debug")
        optimize "Off"
    filter {"Release"}
        runtime "Release"
        targetname ("unittest")
        optimize "Full"
    filter{}
