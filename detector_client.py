"""Example client: import this in your Raspberry Pi main program.

    from detector_client import Detector
    det = Detector("192.168.1.50", 5000)   # or "127.0.0.1" if on the same machine
    letter = det.detect()                  # "c", "s", "t" or "n"
"""
import socket


class Detector:
    def __init__(self, host="127.0.0.1", port=5000, timeout=5.0):
        self.sock = socket.create_connection((host, port), timeout=timeout)
        self.f = self.sock.makefile("rw")

    def _ask(self, cmd):
        self.f.write(cmd + "\n")
        self.f.flush()
        return self.f.readline().strip()

    def detect(self):
        return self._ask("DETECT")

    def ping(self):
        return self._ask("PING") == "pong"

    def close(self):
        self.sock.close()


if __name__ == "__main__":
    d = Detector()
    print("alive:", d.ping())
    print("shape:", d.detect())
