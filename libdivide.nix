{ pkgs }:

pkgs.stdenv.mkDerivation rec {
  pname = "libdivide";
  version = "5.3.0"; # или подходящая версия для rocroller

  src = pkgs.fetchFromGitHub {
    owner = "ridiculousfish";
    repo  = "libdivide";
    # подставь нужный тег/коммит
    rev  = "952dceb6a77d505e1c2919526bba2181999e2818";  # пример
    hash = "sha256-duO+4TtTa57MQ1FIShemIPnRkRrQCT4zDGH0/y70j40=";
  };

  nativeBuildInputs = with pkgs; [ cmake ninja ];

  cmakeFlags = [
    "-DCMAKE_INSTALL_PREFIX=${placeholder "out"}"
    "-DLIBDIVIDE_BUILD_TESTS=OFF"
    "-DLIBDIVIDE_BUILD_EXAMPLES=OFF"
  ];

  meta = with pkgs.lib; {
    description = "Optimized integer division library (libdivide)";
    homepage    = "https://github.com/ridiculousfish/libdivide";
    license     = licenses.zlib;
    platforms   = platforms.linux;
  };
}

