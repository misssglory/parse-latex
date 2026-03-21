{
  description = "TensorFlow ROCm env + hipSPARSELt package with separate rocm-cmake derivation";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    utils.url   = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, utils }:
    utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };

        # Импортируем rocm-cmake из отдельного файла
        rocm-cmake = import ./rocm-cmake.nix { inherit pkgs; };
        libdivide  = import ./libdivide.nix  { inherit pkgs; };
        nanobind = import ./nanobind.nix  { inherit pkgs; };

        hipsparselt = pkgs.stdenv.mkDerivation rec {
          pname = "hipsparselt";
          version = "7.2.0";

          src = pkgs.fetchFromGitHub {
            owner = "ROCm";
            repo  = "rocm-libraries";
            rev   = "7d2ed43282dbb125945c08a2441b00fd11e7b962";
            hash  = "sha256-Vdzcj/ZKr8KQC+HMAIhGk1gfx35fD3XHJsT5EYGYMKQ=";
          };

          nativeBuildInputs = with pkgs; [
            cmake
            ninja
            python312
            git
            rocm-cmake
	    yaml-cpp
          ];

          buildInputs = with pkgs; [
            rocmPackages.hipsparse
            rocmPackages.rocblas
            rocmPackages.rocsparse
            rocmPackages.rocm-core
            # rocmPackages.hip-common
            rocmPackages.clr
            rocmPackages.roctracer
            rocmPackages.amdsmi
            #rocmPackages.hipfort
            #haskellPackages.hip
	    rocmPackages.llvm.llvm
	    rocmPackages.llvm.lld
	    rocmPackages.hipcc
            #flang
	    gfortran
	    msgpack-cxx
            yaml-cpp
            fmt
            spdlog
            libdivide
	    gtest
	    blas
	    cli11
	    # nanobind
	    python312
	    python312Packages.nanobind
          ];

          # ВАЖНО: CMAKE_MODULE_PATH должен указывать на реальный путь из rocm-cmake.
          # После первой сборки посмотри `ls -R $(nix build .#rocm-cmake --no-link -L | tail -1)`,
          # и подставь сюда правильный подкаталог (пример для Arch):
          #   ${rocm-cmake}/share/rocmcmakebuildtools/cmake
          cmakeFlags = [
            "-DROCM_PATH=${pkgs.rocmPackages.rocm-core}"
            "-DCMAKE_MODULE_PATH=${rocm-cmake}/share/rocmcmakebuildtools/cmake"
#    "-DCMAKE_INSTALL_PREFIX=${placeholder "out"}"

 #   "-DCMAKE_PREFIX_PATH=${pkgs.rocmPackages.clr}"
  #  "-DROCROLLER_BUILD_TESTING=OFF"   # avoid extra yaml work[web:145]
    "-DROCROLLER_YAML_BACKEND=YAML_CPP"   # prefer LLVM yaml over yaml-cpp[web:121]
    "-DROCROLLER_ENABLE_GEMM_CLIENT_TESTS=OFF"
    "-DROCROLLER_ENABLE_CATCH=OFF"
      "-DCMAKE_CXX_COMPILER=${pkgs.rocmPackages.hipcc}/bin/hipcc"
  "-DCMAKE_C_COMPILER=${pkgs.rocmPackages.hipcc}/bin/hipcc"
  # HIP settings
  "-DHIP_PATH=${pkgs.rocmPackages.clr}"
  "-DHIP_ROOT_DIR=${pkgs.rocmPackages.clr}"
  "-DHIP_PLATFORM=amd"
  "-DHIP_COMPILER=clang"
  # ROCm settings
  "-DAMDGPU_TARGETS=gfx1100;gfx1101;gfx1102;gfx1103"  # For RDNA3 (8845HS)
          ];

          # На первом шаге можно попробовать без патчинга FetchContent и посмотреть,
          # станет ли hipSPARSELt/hipblas-common использовать уже установленный rocm-cmake.
          # При необходимости потом добавим postPatch, чтобы отключить сетевые FetchContent.


postPatch = ''
  # Disable FetchContent for various dependencies
  substituteInPlace shared/rocroller/CMakeLists.txt \
    --replace "FetchContent_MakeAvailable(fmt)" \
              "find_package(fmt CONFIG REQUIRED)"
  
  substituteInPlace shared/rocroller/CMakeLists.txt \
    --replace "FetchContent_MakeAvailable(spdlog)" \
              "find_package(spdlog CONFIG REQUIRED)"
  
  substituteInPlace shared/rocroller/CMakeLists.txt \
    --replace "FetchContent_MakeAvailable(libdivide)" \
              "find_package(libdivide CONFIG REQUIRED)"
  
  substituteInPlace shared/rocroller/CMakeLists.txt \
    --replace "FetchContent_MakeAvailable(yaml_cpp)" \
              "find_package(yaml-cpp CONFIG REQUIRED)"
  
  substituteInPlace shared/rocroller/CMakeLists.txt \
    --replace "FetchContent_MakeAvailable(googletest)" \
              "find_package(GTest CONFIG REQUIRED)"
  
  # Fix GPUArchitectureGenerator linking
  substituteInPlace shared/rocroller/GPUArchitectureGenerator/CMakeLists.txt \
    --replace "yaml-cpp::yaml-cpp" "yaml-cpp"
  
  # Fix nanobind in tensilelite
  substituteInPlace projects/hipblaslt/tensilelite/rocisa/CMakeLists.txt \
    --replace "FetchContent_MakeAvailable(nanobind)" \
              "# Nix: using system nanobind"
  
  sed -i '1i\
  # Find nanobind Python module\
  execute_process(\
    COMMAND python -c "import nanobind; print(nanobind.__file__)"\
    OUTPUT_STRIP_TRAILING_WHITESPACE\
    OUTPUT_VARIABLE NANOBIND_MODULE_PATH\
    ERROR_VARIABLE NANOBIND_IMPORT_ERROR\
    RESULT_VARIABLE NANOBIND_IMPORT_RESULT\
  )\
  \
  message(STATUS "Checking nanobind Python module...")\
  message(STATUS "  Import result: ''${NANOBIND_IMPORT_RESULT}")\
  if(NANOBIND_IMPORT_RESULT)\
    message(STATUS "  Import error: ''${NANOBIND_IMPORT_ERROR}")\
  endif()\
  message(STATUS "  nanobind module path: ''${NANOBIND_MODULE_PATH}")\
  \
  execute_process(\
    COMMAND python -m nanobind --cmake_dir\
    OUTPUT_STRIP_TRAILING_WHITESPACE\
    OUTPUT_VARIABLE NANOBIND_CMAKE_DIR\
    ERROR_VARIABLE NANOBIND_ERROR\
    RESULT_VARIABLE NANOBIND_RESULT\
  )\
  \
  message(STATUS "Finding nanobind CMake dir...")\
  message(STATUS "  Command result: ''${NANOBIND_RESULT}")\
  if(NANOBIND_RESULT)\
    message(STATUS "  Error output: ''${NANOBIND_ERROR}")\
  endif()\
  message(STATUS "  nanobind CMake dir: ''${NANOBIND_CMAKE_DIR}")\
  \
  if(EXISTS "''${NANOBIND_CMAKE_DIR}")\
    message(STATUS "  ✓ Directory exists: ''${NANOBIND_CMAKE_DIR}")\
  else()\
    message(WARNING "  ✗ Directory does NOT exist: ''${NANOBIND_CMAKE_DIR}")\
  endif()\
  \
  list(APPEND CMAKE_PREFIX_PATH "''${NANOBIND_CMAKE_DIR}")\
  message(STATUS "Updated CMAKE_PREFIX_PATH: ''${CMAKE_PREFIX_PATH}")\
  \
  find_package(nanobind CONFIG REQUIRED)\
  if(nanobind_FOUND)\
    message(STATUS "✓ nanobind found! Version: ''${nanobind_VERSION}")\
    message(STATUS "  nanobind include dir: ''${nanobind_INCLUDE_DIR}")\
    message(STATUS "  nanobind lib dir: ''${nanobind_LIB_DIR}")\
  endif()\
  ' projects/hipblaslt/tensilelite/rocisa/CMakeLists.txt

'';


          buildPhase = ''
            cd projects/hipsparselt
            mkdir -p build/release
            cd build/release
            cmake ../.. \
              -DCMAKE_INSTALL_PREFIX=$out
            make -j"$(nproc)"
          '';

          installPhase = ''
            cd projects/hipsparselt/build/release
            make install
          '';

          outputs = [ "out" ];

          meta = with pkgs.lib; {
            description = "ROCm hipSPARSELt sparse marshalling library (using packaged rocm-cmake)";
            homepage    = "https://rocm.docs.amd.com/projects/hipSPARSELt/en/latest/install/install-hipsparselt.html";
            license     = licenses.mit;
            platforms   = platforms.linux;
          };
        };

        rocmLibs = with pkgs.rocmPackages; [
          clr
          hipblas
          hipblaslt
          miopen
          rccl
          rocblas
          rocsolver
          rocsparse
          rocm-smi
          hsakmt
          rocm-core
          hipsparse
          clr.icd
        ] ++ [
          hipsparselt
        ];

        runtimeLibs = with pkgs; [
          stdenv.cc.cc.lib
          zlib
          glibc
          libgcc.lib
          xorg.libX11
          libGL
          glib
        ] ++ rocmLibs;

        ldLibPath = pkgs.lib.makeLibraryPath runtimeLibs;

      in
      {
        # Отдельно можно экспортировать rocm-cmake как пакет
        packages.rocm-cmake = rocm-cmake;
        packages.libdivide = libdivide;
        packages.hipsparselt = hipsparselt;
        packages.default = hipsparselt;

        devShells.default = pkgs.mkShell {
          buildInputs =
            with pkgs; [
              uv
              python312
              rocmPackages.rocm-core
              rocmPackages.hsakmt
            ] ++ rocmLibs;

          NIX_LD_LIBRARY_PATH = ldLibPath;
          NIX_LD = pkgs.lib.fileContents "${pkgs.stdenv.cc}/nix-support/dynamic-linker";

          shellHook = ''
            ORIGINAL_LD_LIBRARY_PATH="$LD_LIBRARY_PATH"

            export HSA_OVERRIDE_GFX_VERSION=11.0.2

            if [ ! -d ".venv" ]; then
              echo "Creating virtual environment..."
              uv venv
            fi
            . .venv/bin/activate

            echo "Installing/Updating dependencies..."
            uv pip install \
              "keras>=3.0.0" \
              "https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/tensorflow_rocm-2.20.0.dev0%2Bselfbuilt-cp312-cp312-manylinux_2_28_x86_64.whl" \
              "numpy" \
              "matplotlib" \
              "loguru"

            export LD_LIBRARY_PATH="${ldLibPath}"
            if [ -n "$ORIGINAL_LD_LIBRARY_PATH" ]; then
              export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$ORIGINAL_LD_LIBRARY_PATH"
            fi

            echo "Environment ready! GPU: 8845HS (RDNA3)"
          '';
        };
      }
    );
}

