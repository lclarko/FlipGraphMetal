"""Independent byte-level oracle for FGM-CONTRACT-v1; no live normalization."""
import hashlib
import itertools
import struct

DOMAINS = {'ZT': 1, 'F2': 2}
U32 = (1 << 32)-1
U64 = (1 << 64)-1


def unsigned(value, maximum):
    if type(value) is not int or not 0 <= value <= maximum:
        raise ValueError('unsigned field outside encoding range')
    return value


def validated(data):
    if data['domain'] not in DOMAINS or data['orientation'] != 'cyclic-w':
        raise ValueError('unsupported domain or orientation')
    dims = tuple(unsigned(v, U32) for v in data['dimensions'])
    if len(dims) != 3 or not all(dims):
        raise ValueError('expected three positive dimensions')
    rank = unsigned(data['rank'], U32)
    if not rank:
        raise ValueError('rank must be positive')
    widths = tuple(dims[i]*dims[(i+1)%3] for i in range(3))
    for width in widths:
        unsigned(width, U32)
    factors = [data[key] for key in 'uvw']
    allowed = (-1,0,1) if data['domain'] == 'ZT' else (0,1)
    for rows, width in zip(factors, widths):
        if len(rows) != rank:
            raise ValueError('factor rank mismatch')
        for row in rows:
            if len(row) != width or any(type(v) is not int or v not in allowed for v in row):
                raise ValueError('coefficient shape or domain mismatch')
    return dims, rank, factors


def encoded(data, canonical):
    dims, rank, factors = validated(data)
    terms = []
    gauges = ((1,1,1),(-1,-1,1),(-1,1,-1),(1,-1,-1))
    for r in range(rank):
        parts = [factor[r] for factor in factors]
        if canonical and data['domain'] == 'ZT':
            term = max(tuple(value*sign for part, sign in zip(parts, gauge) for value in part)
                       for gauge in gauges)
        else:
            term = tuple(itertools.chain.from_iterable(parts))
        terms.append(term)
    if canonical:
        terms.sort()
    magic = b'FGMSCHEME\0' if canonical else b'FGMFACTORS\0'
    header = struct.pack('<HBBIIII', 1, DOMAINS[data['domain']], 1, *dims, rank)
    return magic + header + bytes(value & 255 for term in terms for value in term)


def identity(data, canonical=True):
    prefix = 'fgm-scheme-v1:' if canonical else 'fgm-factors-v1:'
    return prefix + hashlib.sha256(encoded(data, canonical)).hexdigest()


def lp32(value):
    if not isinstance(value, str):
        raise ValueError('expected UTF-8 text')
    payload = value.encode('utf-8', errors='strict')
    unsigned(len(payload), U32)
    return struct.pack('<I', len(payload)) + payload


def selection_bytes(seed, namespace, presentation_id):
    return (b'FGMSELECT\0' + struct.pack('<HQ', 1, unsigned(seed,U64))
            + lp32('presentation') + lp32(namespace) + lp32(presentation_id))


def selection_key(seed, namespace, presentation_id):
    return (hashlib.sha256(selection_bytes(seed,namespace,presentation_id)).digest(),
            presentation_id.encode('utf-8'))


def schoolbook(n, domain='ZT'):
    a,b,c=n
    terms=[]
    for i,k,j in itertools.product(range(a),range(b),range(c)):
        u,v,w=[0]*(a*b),[0]*(b*c),[0]*(c*a)
        u[i*b+k]=v[k*c+j]=w[j*a+i]=1
        terms.append((u,v,w))
    return dict(domain=domain,orientation='cyclic-w',dimensions=list(n),rank=len(terms),
                **{key:[t[p] for t in terms] for p,key in enumerate('uvw')})
