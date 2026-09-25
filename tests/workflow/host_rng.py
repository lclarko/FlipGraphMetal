"""Independent MT19937-64 and bounded host selection for FGM-CONTRACT-v1."""
U64=(1<<64)-1


def checked(value,maximum=U64):
    if type(value) is not int or not 0<=value<=maximum:
        raise ValueError('unsigned field outside range')
    return value


class HostRNG:
    def __init__(self,seed):
        checked(seed,(1<<32)-1)
        self.words=[seed]
        for i in range(1,312):
            last=self.words[-1]
            self.words.append((6364136223846793005*(last^(last>>62))+i)&U64)
        self.index=0
        self.draws=0

    def next(self):
        i=self.index
        y=(self.words[i]&0xffffffff80000000)|(self.words[(i+1)%312]&0x7fffffff)
        z=self.words[(i+156)%312]^(y>>1)^(0xb5026f5aa96619e9 if y&1 else 0)
        self.words[i]=z
        self.index=(i+1)%312
        self.draws+=1
        z^=(z>>29)&0x5555555555555555
        z^=(z<<17)&0x71d67fffeda60000
        z^=(z<<37)&0xfff7eee000000000
        return (z^(z>>43))&U64

    def bounded(self,bound):
        checked(bound)
        if not bound:
            raise ValueError('zero selection bound')
        threshold=(1<<64)%bound
        for _ in range(64):
            word=self.next()
            if word>=threshold:
                return word%bound
        raise RuntimeError('host selection draw budget exhausted')

    def select(self,weights):
        if not weights:
            raise ValueError('no eligible parents')
        total=0
        for weight in weights:
            total=checked(total+checked(weight))
        if not total:
            return self.bounded(len(weights))
        sample=self.bounded(total)
        cumulative=0
        for i,weight in enumerate(weights):
            cumulative+=weight
            if cumulative>sample:
                return i
        raise AssertionError('unreachable weighted selection')
