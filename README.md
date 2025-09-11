# RayTracing

This is a ray tracing project implemented in C++ with a GUI. It is heavily inspired
by [Ray Tracing in One Weekend](https://raytracing.github.io/).
I can not recommend the books enough.

![Final Render, Recompressed](FinalRender.png)

It is a CMake project and should be easy to build on any platform. 
It has been tested on Windows and Linux.

The preferred compiler is `clang++` as it seems to generate the best code.

## Dependencies

The package manager of choice is `vcpkg`.
If `VCPKG_ROOT` is set,
the `cmake-debug` and `cmake-release` cmake profiles will automatically resolve dependencies.

The dependencies are:
- `SFML` for the GUI to display the image while it is being rendered.
- `stb` to write the rendered image to a PNG file.

## Building
To build the project, run the following commands:

On Windows (PowerShell, Visual Studio Terminal):
```powershell
cmake --preset cmake-release
cmake --build --preset cmake-release --config Release
```

On a Debian like system:
```bash
sudo apt update && sudo apt upgrade
sudo apt install git gcc g++ zip unzip wget clang cmake build-essential curl tar pkg-config libx11-dev libxi-dev libxrandr-dev libxcursor-dev libxi-dev libudev-dev libgl1-mesa-dev
git clone https://github.com/microsoft/vcpkg.git
git clone https://github.com/amuhak/RayTracing.git
cd vcpkg
./bootstrap-vcpkg.sh
export VCPKG_ROOT=$(pwd)
cd ../RayTracing/
cmake --preset cmake-release -DCMAKE_CXX_COMPILER=/usr/bin/clang++ -DCMAKE_MAKE_PROGRAM=/usr/bin/make
cmake --build --preset cmake-release --config Release -j
./cmake-release/RayTracing
```

Configurations for the GUI and headless rendering are available in `camera.cpp`. NOTE: the GUI does not work well with WSL. Turn it off (`USE_DISPLAY` must be set to `false`).

