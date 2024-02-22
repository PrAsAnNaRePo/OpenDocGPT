import os
import platform
import torch

os.system("git clone https://github.com/ggerganov/llama.cpp.git")

if platform.system() == "Windows":
    os.system("cd llama.cpp && ./start_windows.bat")
    if torch.cuda.is_available():
        os.system("cd llama.cpp && make LLAMA_OPENBLAS=1")
elif platform.system() == "Linux":
    os.system("cd llama.cpp && make")
elif platform.system() == "Darwin":
    os.system("cd llama.cpp && make")
else:
    print("Unsupported OS")

print("Downloading model...")
os.system("wget https://huggingface.co/TheBloke/openchat-3.5-0106-GGUF/resolve/main/openchat-3.5-0106.Q4_K_M.gguf")
