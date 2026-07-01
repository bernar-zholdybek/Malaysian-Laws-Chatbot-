{ pkgs, lib, ... }:

{
  # 1. Enable Python and automatically provision the virtual environment
  languages.python = {
    enable = true;
    venv = {
      enable = true;
      quiet = true;
      # Automatically pip install your packages on shell startup
      requirements = ''
        streamlit
        langchain
        langchain-huggingface
        langchain-chroma
        langchain-google-genai
        langchain-text-splitters
        google-generativeai
        chromadb
        pypdf
        sentence-transformers
      '';
    };
  };

  # 2. System packages required to compile or run C/C++ extensions
  # (e.g., hnswlib for ChromaDB, Tokenizers for Sentence-Transformers)
  packages = with pkgs; [
    stdenv.cc.cc.lib
    zlib
    gcc
    gnumake
    sqlite
  ];

  # 3. Force NixOS to expose critical shared libraries to the pip virtualenv
  env.LD_LIBRARY_PATH = lib.makeLibraryPath (with pkgs; [
    stdenv.cc.cc.lib
    zlib
    sqlite
  ]);

  # 4. A helpful greeting when entering your development shell
  enterShell = ''
    echo "========================================================="
    echo "🚀 LangChain & Streamlit Dev Environment Loaded!"
    echo "📦 All libraries are isolated in your local .devenv/ state."
    echo "========================================================="
    echo "👉 To start your app: streamlit run app.py"
  '';
}
