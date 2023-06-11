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
    
project "unittest"
    kind "ConsoleApp"
    language "C++"
    targetdir "bin/"
    systemversion "latest"
    flags { "MultiProcessorCompile", "NoPCH" }

    -- Src
    files { "unittest.cpp", "voxelization.hpp" }

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

project "main"
    kind "ConsoleApp"
    language "C++"
    targetdir "bin/"
    systemversion "latest"
    flags { "MultiProcessorCompile", "NoPCH" }

    -- Src
    files { "main.cpp", "voxelization.hpp" }

    -- UTF8
    postbuildcommands { 
        "mt.exe -manifest ../utf8.manifest -outputresource:\"$(TargetDir)$(TargetName).exe\" -nologo"
    }

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

    includePrLib()

    symbols "On"

    filter {"Debug"}
        runtime "Debug"
        targetname ("Main_Debug")
        optimize "Off"
    filter {"Release"}
        runtime "Release"
        targetname ("Main")
        optimize "Full"
    filter{}
