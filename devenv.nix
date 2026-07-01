{ pkgs, lib, ... }:

{
  
  languages.python = {
    enable = true;
    venv = {
      enable = true;
      quiet = true;



      # libraries

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



  # system packages

  packages = with pkgs; [
    stdenv.cc.cc.lib
    zlib
    gcc
    gnumake
    sqlite
  ];



  env.LD_LIBRARY_PATH = lib.makeLibraryPath (with pkgs; [
    stdenv.cc.cc.lib
    zlib
    sqlite
  ]);

  
  enterShell = '' echo " To start your app: streamlit run app.py" '';

}
