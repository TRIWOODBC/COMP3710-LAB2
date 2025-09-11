import os
import sys

def run_cmd(cmd):
    print(f"→ {cmd}")
    os.system(cmd)

# 默认 commit 信息
msg = "add DFT AND FFT comparison example"

# 如果运行时给了参数，就用参数作为 commit 信息
if len(sys.argv) > 1:
    msg = " ".join(sys.argv[1:])

# Git 常规操作
run_cmd("git add .")
run_cmd(f'git commit -m "{msg}"')
run_cmd("git push origin main")
