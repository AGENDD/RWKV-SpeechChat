@echo off
REM 运行Python脚本
python main.py --multiturns --rwkv_path model/RWKV-x060-World-3B-v2.1-20240417-ctx4096.pth --weights_path model/rwkv-adapter-speechQA-VoiceAssistant-final.pth

pause