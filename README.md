# RayTracing

This is a ray tracing project implemented in C++ with a GUI. It is heavily inspired
by [Ray Tracing in One Weekend](https://raytracing.github.io/).
I can not recommend the books enough.

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

```bash
cmake --preset cmake-release
cmake --build --preset cmake-release --config Release
```

Configurations for the GUI and headless rendering are available in `camera.cpp`.