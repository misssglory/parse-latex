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
            python3
            git
            rocm-cmake
          ];

          buildInputs = with pkgs; [
            rocmPackages.hipsparse
            rocmPackages.rocblas
            rocmPackages.rocsparse
            rocmPackages.rocm-core
            # rocmPackages.hip-common
            rocmPackages.clr
            #haskellPackages.hip
            msgpack-cxx
            yaml-cpp
            fmt
            spdlog
            libdivide
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
 #   "-DROCROLLER_YAML_BACKEND=LLVM"   # prefer LLVM yaml over yaml-cpp[web:121]
          ];

          # На первом шаге можно попробовать без патчинга FetchContent и посмотреть,
          # станет ли hipSPARSELt/hipblas-common использовать уже установленный rocm-cmake.
          # При необходимости потом добавим postPatch, чтобы отключить сетевые FetchContent.

postPatch = ''
  # уже существующие штуки (rocm-cmake, hipblas-common и т.п.) если есть…

  # Отключаем FetchContent для fmt в rocroller
  substituteInPlace shared/rocroller/CMakeLists.txt \
    --replace "FetchContent_MakeAvailable(fmt)" \
              "# Nix: use system fmt instead of FetchContent_MakeAvailable(fmt)"

  # Отключаем FetchContent для spdlog в rocroller
  substituteInPlace shared/rocroller/CMakeLists.txt \
    --replace "FetchContent_MakeAvailable(spdlog)" \
              "# Nix: use system spdlog instead of FetchContent_MakeAvailable(spdlog)"

  # Отключаем FetchContent для libdivide в rocroller
  substituteInPlace shared/rocroller/CMakeLists.txt \
    --replace "FetchContent_MakeAvailable(libdivide)" \
              "# Nix: use system libdivide instead of FetchContent_MakeAvailable(libdivide)"

  # Если нужно, можно добавить find_package или include_directories,
  # но rocroller, скорее всего, просто добавляет libdivide как интерфейсный include.

  # Подстрахуемся: явно потребуем system fmt и spdlog через find_package
  sed -i '1i find_package(fmt CONFIG REQUIRED)' shared/rocroller/CMakeLists.txt
  sed -i '2i find_package(spdlog CONFIG REQUIRED)' shared/rocroller/CMakeLists.txt
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

