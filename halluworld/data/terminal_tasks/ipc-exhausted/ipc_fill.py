"""Fill SysV message queue until sends fail (benchmark setup). Linux only."""
import errno
import sys

import sysv_ipc

KEY = 0xCAFEBABE

try:
    mq = sysv_ipc.MessageQueue(KEY, sysv_ipc.IPC_CREAT, int("0666", 8))
except TypeError:
    mq = sysv_ipc.MessageQueue(KEY, sysv_ipc.IPC_CREAT)

n = 0
while n < 50000:
    try:
        mq.send(b"\x01" * 4, block=False)
    except sysv_ipc.BusyError:
        break
    except OSError as e:
        if e.errno in (errno.EAGAIN, errno.ENOMSG, 11):
            break
        raise
    n += 1
