"""Producer that fails when the benchmark queue is full."""
import errno
import sys

import sysv_ipc

KEY = 0xCAFEBABE


def main() -> int:
    try:
        mq = sysv_ipc.MessageQueue(KEY)
    except sysv_ipc.ExistentialError:
        print("queue missing", file=sys.stderr)
        return 1
    try:
        mq.send(b"!", block=False)
    except sysv_ipc.BusyError:
        print("No space left on device (message queue is full)", file=sys.stderr)
        return 2
    except OSError as e:
        if e.errno in (errno.EAGAIN, 11):
            print("No space left on device (message queue is full)", file=sys.stderr)
            return 2
        raise
    print("send-ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
