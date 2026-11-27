"""Seekable file object over HTTP range requests, so zipfile can list/extract remotely."""
import io, urllib.request

class HttpRangeFile(io.RawIOBase):
    def __init__(self, url):
        self.url, self.pos = url, 0
        self.size = int(urllib.request.urlopen(urllib.request.Request(url, method="HEAD")).headers["Content-Length"])
    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos
    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos
    def read(self, n=-1):
        if n < 0 or self.pos + n > self.size: n = self.size - self.pos
        if n <= 0: return b""
        r = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{self.pos+n-1}"})
        d = urllib.request.urlopen(r, timeout=120).read(); self.pos += len(d); return d
    def readinto(self, b):
        d = self.read(len(b)); b[:len(d)] = d; return len(d)
