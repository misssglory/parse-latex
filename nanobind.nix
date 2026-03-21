{ pkgs }:

pkgs.stdenv.mkDerivation rec {
  pname = "nanobind";
  version = "2.6.1";

  src = pkgs.fetchFromGitHub {
    owner = "wjakob";
    repo  = "nanobind";
    rev   = "9b3afa9dbdc23641daf26fadef7743e7127ff92f";
    # first run with dummy hash, then copy 'got:' value
    hash  = "sha256-REPLACE_ME";
  };

  nativeBuildInputs = with pkgs; [
    cmake
    ninja
    python312
  ];

  buildInputs = with pkgs; [
    python312
  ];

  buildPhase = ''
    mkdir -p build
    cd build
    cmake .. -DCMAKE_INSTALL_PREFIX=$out
    make -j"$(nproc)"
  '';

  installPhase = ''
    cd build
    make install
  '';

  outputs = [ "out" ];

  meta = with pkgs.lib; {
    description = "nanobind C++/Python binding library";
    homepage    = "https://github.com/wjakob/nanobind";
    license     = licenses.bsd3;
    platforms   = platforms.linux;
  };
}
