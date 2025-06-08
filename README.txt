# Basic setup, to bring conan into the enviroment
./setup.sh
source ./venv/bin/activate

# Use information from conanfile.py & example_profile
# to get requirements, build those which are missing
# using cmake_layout to put stuff in sensible places
conan install . --build=missing -pr ./example_profile

# Use preset that was generated to configure cmake
cmake --preset conan-release
# if you want look at build/Release/generators/CMakePresets.json to see what is being done for you

# Use the preset to do the cmake build
cmake --build build --preset conan-release

# We could do extra stuff like test/install, but for now just run the built binary
./build/Release/hello
