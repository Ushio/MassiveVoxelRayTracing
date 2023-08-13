import os
import shutil

packageDir = "RTCampPackage"

if not os.path.exists(packageDir):
    os.makedirs(packageDir)
# if not os.path.exists(os.path.join(packageDir, "bin")):
#     os.makedirs(os.path.join(packageDir, "bin"))

srcDir = "."
for file in os.listdir(srcDir):
    if any( file.endswith(ext) for ext in { "cu", "hpp" } ):
        source_path = os.path.join(srcDir, file)
        target_path = os.path.join(packageDir, file)
        shutil.copy2( source_path, target_path )

binFiles = {
    "RTCamp.exe",
    "output.abc",

    # AMD Option
    "hiprtc0505.dll",
    "hiprtc-builtins0505.dll",
    "amd_comgr0505.dll",

    # HDRI
    "monks_forest_2k.hdr",
    "monks_forest_2k_primary.hdr",
}

for file in binFiles:
    shutil.copy2( os.path.join( "bin", file ), os.path.join( packageDir, file ) )

otherFiles = {
    "fps.txt",
    "renderer_introduction.pptx",
    "run.ps1",
    "run.py",
}
for file in otherFiles:
    shutil.copy2( os.path.join( "usecase2_submission", file ), os.path.join( packageDir, file ) )

# CUDA
cuda_path = os.environ.get('CUDA_PATH')
shutil.copy2( os.path.join( cuda_path, "bin/nvrtc64_120_0.dll"), os.path.join( packageDir, "nvrtc64_120_0.dll" ))
shutil.copy2( os.path.join( cuda_path, "bin/nvrtc-builtins64_120.dll"), os.path.join( packageDir, "nvrtc-builtins64_120.dll" ) )