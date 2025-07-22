@echo off
echo Activating LangChain environment...
call langchain_env\Scripts\activate.bat
echo LangChain environment activated!
echo Available packages: langchain, langchain-openai, openai (v1.97.0), tiktoken (v0.9.0)
cmd /k
