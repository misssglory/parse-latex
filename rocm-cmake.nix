{ pkgs }:

pkgs.stdenv.mkDerivation rec {
  pname = "rocm-cmake";
  version = "7.2.0";

  src = pkgs.fetchFromGitHub {
    owner = "ROCm";
    repo  = "rocm-cmake";
    # Лучше использовать тег/коммит под твою версию ROCm
    rev  = "c01b4f1fd36a94d26c76e7f617b57577b3b84275";
    hash = "sha256-xE855XZzvrtBoR5iruIDXYiRAjaZNbCEX2zWug9sehA=";
  };

  nativeBuildInputs = with pkgs; [
    cmake
    ninja
    python3
  ];

  # Стандартный cmake-паттерн: build/ + install
  # cmake hooks в Nix это сделают автоматически

  cmakeFlags = [
    "-DCMAKE_INSTALL_PREFIX=${placeholder "out"}"
  ];

  meta = with pkgs.lib; {
    description = "AMD ROCm CMake modules (rocm-cmake)";
    homepage    = "https://github.com/ROCm/rocm-cmake";
    license     = licenses.mit;
    platforms   = platforms.linux;
  };
}

