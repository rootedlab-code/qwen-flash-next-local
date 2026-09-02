"""Print the header / tensor-info section of a GGUF, even a truncated one.

  gguf-header.py <file.gguf>

Reads only the first 256 MiB, so it works on a 45 GB shard and on the saved
header of one, and it never pulls tensor data into the page cache. This is how
the placement table in the README was produced: every tensor's name, shape,
quantization type and size in bytes, without loading the model.

Needs llama.cpp's gguf-py on PYTHONPATH (or a pip-installed `gguf`).
"""
import struct, sys
from gguf.constants import GGML_QUANT_SIZES, GGMLQuantizationType

U8,I8,U16,I16,U32,I32,F32,BOOL,STR,ARR,U64,I64,F64 = range(13)
FIXED = {U8:('B',1), I8:('b',1), U16:('H',2), I16:('h',2), U32:('I',4),
         I32:('i',4), F32:('f',4), BOOL:('?',1), U64:('Q',8), I64:('q',8), F64:('d',8)}
PREFIX_BYTES = 256 << 20   # header + tensor-info only: never read a 44 GB gguf into RAM

class R:
    def __init__(self, buf): self.b, self.o = buf, 0
    def raw(self, n):
        v = self.b[self.o:self.o+n]
        if len(v) < n: raise EOFError
        self.o += n
        return v
    def u32(self): return struct.unpack('<I', self.raw(4))[0]
    def u64(self): return struct.unpack('<Q', self.raw(8))[0]
    def s(self):   return self.raw(self.u64()).decode('utf-8', 'replace')
    def val(self, t):
        if t in FIXED:
            f, n = FIXED[t]
            return struct.unpack('<'+f, self.raw(n))[0]
        if t == STR: return self.s()
        if t == ARR:
            et, n = self.u32(), self.u64()
            return [self.val(et) for _ in range(n)]
        raise ValueError(f'unknown value type {t}')

def nbytes(dims, ggml_type):
    n = 1
    for d in dims: n *= d
    blk, tsz = GGML_QUANT_SIZES[ggml_type]
    return n // blk * tsz

def parse(path):
    with open(path, 'rb') as fh:
        r = R(fh.read(PREFIX_BYTES))
    assert r.raw(4) == b'GGUF', 'not a GGUF file'
    r.u32()
    ntensors, nkv = r.u64(), r.u64()
    for _ in range(nkv):
        r.s(); r.val(r.u32())
    out = []
    for _ in range(ntensors):
        try:
            name = r.s()
            dims = [r.u64() for _ in range(r.u32())]
            out.append((name, dims, r.u32(), r.u64()))
        except EOFError:
            print(f'  [header truncated after {len(out)}/{ntensors} tensors]', file=sys.stderr)
            break
    return ntensors, out

if __name__ == '__main__':
    total, tensors = parse(sys.argv[1])
    print(f'tensors declared: {total}, read: {len(tensors)}')
    for name, dims, tt, _ in tensors:
        print(f'{nbytes(dims,tt):>14,}  {GGMLQuantizationType(tt).name:<10} '
              f'{"x".join(map(str,dims)):<24} {name}')
