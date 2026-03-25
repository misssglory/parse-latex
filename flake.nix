{
  description = "Python development environment with VSCodium, uv, TensorFlow, and OpenCV";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.11";
    flake-utils.url = "github:numtide/flake-utils";
    home-manager.url = "github:nix-community/home-manager";
    home-manager.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs = { self, nixpkgs, flake-utils, home-manager }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        
        # Python 3.13
        python = pkgs.python313;
        
        # Create a Python environment with system packages
        pythonWithLibs = pkgs.python313.withPackages (ps: with ps; [
          uv
          pkgs.zeromq
          pkgs.stdenv.cc.cc.lib
        ]);
        
        # Home Manager configuration
        homeConfig = home-manager.lib.homeManagerConfiguration {
          inherit pkgs;
          modules = [
            {
              home = {
                username = builtins.getEnv "USER";
                homeDirectory = builtins.getEnv "HOME";
                stateVersion = "25.11";
              };
              
              programs.vscode = {
                enable = true;
                package = pkgs.vscodium-fhs;
                
                profiles.default.extensions = with pkgs.vscode-extensions; [
                  ms-python.python
                  ms-python.vscode-pylance
                  ms-toolsai.jupyter
                  ms-toolsai.jupyter-keymap
                  ms-toolsai.jupyter-renderers
                  ms-toolsai.vscode-jupyter-cell-tags
                  ms-toolsai.vscode-jupyter-slideshow
                  ms-toolsai.vscode-tensorboard
                  ms-python.black-formatter
                  ms-python.isort
                  charliermarsh.ruff
                  eamodio.gitlens
                  redhat.vscode-yaml
                ];
                
                userSettings = {
                  "python.defaultInterpreterPath" = "./.venv/bin/python";
                  "python.terminal.activateEnvironment" = true;
                  "python.terminal.activateEnvInCurrentTerminal" = true;
                  "python.terminal.launchArgs" = [
                    "--no-warnings"
                  ];
                  "editor.formatOnSave" = true;
                  "editor.defaultFormatter" = "ms-python.black-formatter";
                  "[python]" = {
                    "editor.formatOnSave" = true;
                    "editor.codeActionsOnSave" = {
                      "source.organizeImports" = true;
                    };
                    "editor.defaultFormatter" = "ms-python.black-formatter";
                  };
                  "black-formatter.args" = ["--line-length" "88"];
                  "black-formatter.importStrategy" = "fromEnvironment";
                  "ruff.enable" = true;
                  "jupyter.alwaysTrustNotebooks" = true;
                };
              };
            }
          ];
        };
      in
      {
        devShell = pkgs.mkShell {
          buildInputs = with pkgs; [
            pythonWithLibs
            uv
            # Use opencv4 instead of opencv for better compatibility
            opencv4
            gtk3
            pkg-config
            git
            gcc
            stdenv.cc.cc.lib
            zeromq
            libsodium
            libffi
            openssl
            libGL
            libGLU
            libxcb
            # xorg.libX11
            # xorg.libXext
            # xorg.libXrender
            # xorg.libX11
            # Additional dependencies for OpenCV
            gst_all_1.gstreamer
            gst_all_1.gst-plugins-base
            gst_all_1.gst-plugins-good
            gst_all_1.gst-plugins-bad
            gst_all_1.gst-plugins-ugly
          ];
          
          shellHook = ''
            # Set up library paths
            export LD_LIBRARY_PATH=${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.zeromq}/lib:${pkgs.libsodium}/lib:${pkgs.libffi}/lib:${pkgs.openssl}/lib:${pkgs.opencv4}/lib:$LD_LIBRARY_PATH
            
            # Set up OpenCV paths
            export PKG_CONFIG_PATH=${pkgs.opencv4}/lib/pkgconfig:$PKG_CONFIG_PATH
            export PYTHONPATH=$PYTHONPATH:${pkgs.opencv4}/lib/python3.13/site-packages
            
            # Create and activate uv virtual environment
            if [ ! -d ".venv" ]; then
              echo "Creating uv virtual environment with Python 3.13..."
              uv venv --python ${python}/bin/python .venv
            fi
            
            # Activate virtual environment
            source .venv/bin/activate
            
            # Install base packages
            echo "Installing base packages..."
            uv pip install wheel setuptools pip --upgrade
            
            # Install pyzmq and jupyter dependencies
            echo "Installing Jupyter dependencies..."
            uv pip install pyzmq jupyter-client ipykernel
            
            # Install OpenCV and TensorFlow
            echo "Installing OpenCV and TensorFlow..."
            # Use opencv-python-headless to avoid GUI-related issues
            uv pip install opencv-python-headless opencv-contrib-python-headless
	    uv pip install tensorflow_cpu-2.21.0-cp313-cp313-linux_x86_64.whl
            uv pip install numpy matplotlib
            
            # Install remaining packages
            echo "Installing additional packages..."
            uv pip install \
              jupyter \
              jupyterlab \
              ipywidgets \
              black \
              isort \
              ruff \
              pylint \
              mypy \
              pytest \
              pandas \
              scikit-learn \
              seaborn \
	      loguru \
	      tqdm
            
            # Register the kernel for Jupyter
            echo "Registering Jupyter kernel..."
            KERNEL_DIR="$HOME/.local/share/jupyter/kernels/venv"
            mkdir -p "$KERNEL_DIR"
            
            cat > "$KERNEL_DIR/kernel.json" << EOF
{
  "argv": [
    "$(which python)",
    "-m",
    "ipykernel_launcher",
    "-f",
    "{connection_file}"
  ],
  "display_name": "Python 3.13 (venv)",
  "language": "python",
  "env": {
    "LD_LIBRARY_PATH": "${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.zeromq}/lib:$LD_LIBRARY_PATH",
    "PYTHONPATH": "$PWD/.venv/lib/python3.13/site-packages:${pkgs.opencv4}/lib/python3.13/site-packages"
  }
}
EOF
            
            # Set up VSCodium configuration
            export VSCODE_IPC_HOOK_CLI="/run/user/$UID/vscodium-ipc"
            
            # Create project-specific settings
            mkdir -p .vscode
            cat > .vscode/settings.json << EOF
{
  "python.defaultInterpreterPath": "''${workspaceFolder}/.venv/bin/python",
  "python.terminal.activateEnvironment": true,
  "python.terminal.activateEnvInCurrentTerminal": true,
  "python.linting.enabled": true,
  "python.linting.ruffEnabled": true,
  "python.linting.pylintEnabled": false,
  "python.formatting.provider": "black",
  "python.formatting.blackArgs": ["--line-length", "88"],
  "editor.formatOnSave": true,
  "editor.codeActionsOnSave": {
    "source.organizeImports": true
  },
  "python.sortImports.args": ["--profile", "black"],
  "python.envFile": "''${workspaceFolder}/.env",
  "python.analysis.extraPaths": ["${pkgs.opencv4}/lib/python3.13/site-packages"]
}
EOF
            
            # Create .env file with library paths
            cat > .env << EOF
LD_LIBRARY_PATH=${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.zeromq}/lib:${pkgs.libsodium}/lib:${pkgs.libxcb}/lib:$LD_LIBRARY_PATH

PYTHONPATH=${pkgs.opencv4}/lib/python3.13/site-packages
EOF
           export LD_LIBRARY_PATH=${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.zeromq}/lib:${pkgs.libsodium}/lib:${pkgs.libffi}/lib:${pkgs.openssl}/lib:${pkgs.opencv4}/lib:${pkgs.libGL}/lib:$LD_LIBRARY_PATH
            
            echo ""
            echo "╔════════════════════════════════════════════════════════════════╗"
            echo "║  Development environment ready!                                ║"
            echo "║                                                                ║"
            echo "║  • Python: $(python --version)                                 ║"
            echo "║  • Virtual env: .venv (activated)                              ║"
            echo "║  • Python path: $(which python)                                ║"
            echo "║  • OpenCV (headless): installed                                ║"
            echo "║  • TensorFlow: installed                                       ║"
            echo "║                                                                ║"
            echo "║  Testing imports:                                              ║"
            $(python -c "import cv2; print('  ✓ OpenCV:', cv2.__version__)" 2>/dev/null || echo "  ⚠ OpenCV import failed")
            $(python -c "import tensorflow as tf; print('  ✓ TensorFlow:', tf.__version__)" 2>/dev/null || echo "  ⚠ TensorFlow import failed")
            $(python -c "import numpy; print('  ✓ NumPy:', numpy.__version__)" 2>/dev/null || echo "  ⚠ NumPy import failed")
            echo "║                                                                ║"
            echo "║  Run 'codium' to start VSCodium                               ║"
            echo "╚════════════════════════════════════════════════════════════════╝"
            echo ""
          '';
        };
      });
}
