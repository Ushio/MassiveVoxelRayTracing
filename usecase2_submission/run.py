import sys
import subprocess
import paramiko
import time

inst_addr0 = sys.argv[1]
inst_addr1 = sys.argv[2]
key_path = R'C:\Users\Administrator\.ssh\id_rsa'

# ノンブロッキング、つまりローカルのコマンドの終了は待たない。
local_commands = R'RTCamp --frame-range 0 156'
local_process = subprocess.Popen(local_commands.split(' '))

ssh = paramiko.SSHClient()
key = paramiko.RSAKey(filename=key_path)
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(inst_addr1, username='administrator', pkey=key)

# ノンブロッキング、つまりリモートのコマンドの終了は待たない。
remote_commands = R"""
cd $home\usecase2_submission
.\RTCamp --frame-range 156 240
"""
(_, remote_stdout, remote_stderr) = ssh.exec_command(remote_commands)

local_process.wait()

# リモートのログを出力。
print(remote_stdout.read().decode())
print(remote_stderr.read().decode(), file=sys.stderr)

ssh.close()
