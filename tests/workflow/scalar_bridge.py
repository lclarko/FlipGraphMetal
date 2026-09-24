"""Host-only native pinned-arithmetic adapter and independent policy bridge.

Compilation is explicit, isolated and receipted. This does not freeze reviewed
golden traces or establish GPU equivalence. Policy logic stays in reference_policy.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import importlib.util
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

from reference_policy import ARITHMETIC_REFERENCE, ArithmeticEvent, U32, WorkerRNG

ROOT = Path(__file__).resolve().parents[2]
HEADERS = ('core.h', 'addition.h', 'flip_set.h', 'scheme_integer.h', 'scheme_z2.h')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def replace_once(text, before, after):
    if text.count(before) != 1:
        raise ValueError('pinned instrumentation anchor is not unique: ' + before)
    return text.replace(before, after, 1)


def instrument(name, data):
    text = data.decode('utf8')
    changes = []
    if name == 'core.h':
        text = replace_once(text, 'inline uint32_t randomWord(LOCAL RandomState *state) {',
                            'inline uint32_t randomWord(LOCAL RandomState *state) {\n    ++reference_draws;')
        text = replace_once(text, '    state->value = x;\n    return x;',
                            '    state->value = x;\n    reference_words.push_back(x);\n    return x;')
        changes += ['count randomWord invocations', 'record each returned RNG word']
    if name in ('scheme_integer.h', 'scheme_z2.h'):
        text = replace_once(text, '\nprivate:\n', '\npublic: // reference-only primitive access\n')
        klass = 'SchemeInteger' if name == 'scheme_integer.h' else 'SchemeZ2'
        pattern = r'(void ' + klass + r'::addTriplet\([^\n]+\) LOCAL_METHOD \{)'
        if len(re.findall(pattern, text)) != 1:
            raise ValueError('pinned addTriplet anchor is not unique')
        text = re.sub(pattern, r'\1\n    ++reference_appends;', text, count=1)
        changes += ['private primitive access', 'count committed triplet appends']
    return text.encode(), changes


def build(directory, compiler='clang++'):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    snapshot = directory / 'snapshot'
    snapshot.mkdir()
    source = Path(__file__).with_name('scalar_reference.cpp')
    copied_source = directory / source.name
    copied_source.write_bytes(source.read_bytes())
    manifest = {'complete': False, 'arithmetic_reference': ARITHMETIC_REFERENCE,
                'kind': 'host-only-pinned-scalar-adapter', 'headers': {},
                'adapter_sha256': digest(copied_source.read_bytes()),
                'bridge_sha256': digest(Path(__file__).read_bytes())}
    receipt = directory / 'receipt.json'
    def save():
        receipt.write_text(json.dumps(manifest, indent=2) + '\n')
    save()
    try:
        for name in HEADERS:
            data = subprocess.run(['git', '-C', str(ROOT), 'show',
                                   f'{ARITHMETIC_REFERENCE}:src/metal/{name}'],
                                  check=True, capture_output=True, timeout=10).stdout
            transformed, changes = instrument(name, data)
            (snapshot / name).write_bytes(transformed)
            manifest['headers'][name] = {'original_sha256': digest(data),
                                         'instrumented_sha256': digest(transformed),
                                         'transformations': changes}
        executable = directory / 'scalar_reference'
        compiler_path = shutil.which(compiler)
        if compiler_path is None:
            raise ValueError('compiler unavailable: ' + compiler)
        command = [compiler_path, '-std=c++17', '-O2', '-I' + str(snapshot),
                   str(copied_source), '-o', str(executable)]
        manifest['command'] = command
        manifest['compiler_version'] = subprocess.run([compiler_path, '--version'],
            capture_output=True, text=True, check=True, timeout=10).stdout
        save()
        with (directory / 'compile.log').open('xb') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=45)
        if result.returncode:
            raise RuntimeError('native reference compilation failed; see ' + str(directory / 'compile.log'))
        manifest['executable_sha256'] = digest(executable.read_bytes())
        manifest['complete'] = True
        save()
        return executable
    except Exception as error:
        manifest['error'] = str(error)
        save()
        raise


def check_build(binary):
    binary = Path(binary).resolve(strict=True)
    manifest = json.loads((binary.parent / 'receipt.json').read_text())
    if manifest.get('complete') is not True or manifest.get('arithmetic_reference') != ARITHMETIC_REFERENCE:
        raise ValueError('incomplete or wrong-reference scalar build')
    if digest(binary.read_bytes()) != manifest['executable_sha256']:
        raise ValueError('scalar executable changed')
    if digest((binary.parent / 'scalar_reference.cpp').read_bytes()) != manifest['adapter_sha256']:
        raise ValueError('scalar adapter snapshot changed')
    for name, entry in manifest['headers'].items():
        if digest((binary.parent / 'snapshot' / name).read_bytes()) != entry['instrumented_sha256']:
            raise ValueError('scalar header snapshot changed')
    return binary


def validate_rng(initial, result):
    """Check every native word against independent Python xorshift replay."""
    replay = WorkerRNG(0, 0)
    replay.state = initial
    words = result.get('words')
    if not isinstance(words, list) or type(result.get('draws')) is not int or result['draws'] != len(words):
        raise ValueError('invalid native draw record')
    for word in words:
        if type(word) is not int or word != replay.next():
            raise ValueError('native RNG word differs from independent replay')
    if type(result.get('rng')) is not int or result['rng'] != replay.state:
        raise ValueError('native RNG state differs from independent replay')


def operation(binary, scheme, op, rng, ceiling, candidates=None, proposal_limit=None):
    if type(scheme.get('z2')) is not bool:
        raise ValueError('explicit domain required')
    if type(rng) is not int or not 1 <= rng <= U32:
        raise ValueError('nonzero uint32 RNG state required')
    if op not in ('inspect', 'flip', 'reduce', 'plus', 'random', 'existing'):
        raise ValueError('unknown native operation')
    binary = check_build(binary)
    values = [*scheme['n'], scheme['m']]
    for key in 'uvw':
        for row in scheme[key]:
            values.extend(row)
    values.append(int(candidates is not None))
    if candidates is not None:
        if len(candidates) != 3:
            raise ValueError('three candidate lists required')
        for candidate in candidates:
            values.extend([candidate['overflow'], len(candidate['pairs'])])
            for pair in candidate['pairs']:
                values.extend(pair)
    if any(type(value) is not int for value in values):
        raise ValueError('integer input required')
    command = [str(binary), 'F2' if scheme['z2'] else 'ZT', op, str(rng), str(ceiling)]
    if proposal_limit is not None:
        if type(proposal_limit) is not int or not 1 <= proposal_limit <= U32:
            raise ValueError('positive uint32 proposal limit required')
        command.append(str(proposal_limit))
    result = subprocess.run(command, input=' '.join(map(str, values))+'\n',
                            capture_output=True, text=True, timeout=10)
    if result.returncode:
        raise ValueError(result.stderr.strip())
    decoded = json.loads(result.stdout)
    validate_rng(rng, decoded)
    if proposal_limit is not None:
        observations = decoded.get('observations')
        if not isinstance(observations, list) or not 1 <= len(observations) <= proposal_limit:
            raise ValueError('invalid bounded native observation count')
        previous, words = rng, []
        previous_scheme, previous_candidates = scheme, candidates
        rejections = ('tuple_rejection', 'coefficient_rejection')
        for index, item in enumerate(observations):
            if item['operation'] != op or item['rng_before'] != previous:
                raise ValueError('native observation order mismatch')
            observed = item['result']
            validate_rng(previous, observed)
            rejected = observed['outcome'] in rejections
            if index + 1 < len(observations) and not rejected:
                raise ValueError('native invocation continued after a terminal outcome')
            if rejected and (observed['scheme'] != previous_scheme or
                             (previous_candidates is not None and
                              observed['candidates'] != previous_candidates)):
                raise ValueError('rejected native proposal changed ordered state')
            words.extend(observed['words'])
            previous = observed['rng']
            previous_scheme, previous_candidates = observed['scheme'], observed['candidates']
        exhausted = observed['outcome'] in rejections
        expected = 'proposal_exhausted' if exhausted else observed['outcome']
        if exhausted and len(observations) != proposal_limit:
            raise ValueError('native invocation exhausted before its proposal limit')
        if decoded['outcome'] != expected:
            raise ValueError('native invocation outcome differs from its observations')
        for field in ('scheme', 'candidates', 'removed_terms', 'reduction_operations'):
            if decoded[field] != observed[field]:
                raise ValueError('native invocation state differs from its final observation')
        if previous != decoded['rng'] or words != decoded['words']:
            raise ValueError('native invocation/proposal RNG mismatch')
    return decoded


def verify_scheme(scheme):
    """Use the independent dense Python tensor verifier before parent admission."""
    spec = importlib.util.spec_from_file_location('reference_exact_verifier', ROOT / 'tests/metal/verify.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.verify(scheme)


class NativeArithmetic:
    """Policy Scripts-compatible callback backed by actual pinned arithmetic.

    A reference_policy.Policy supplies current RNG and already selected operator.
    Policy replays the returned number of words on its own generator. Schemes and
    ordered candidates persist here. Trace receipts preserve every native result.
    """
    def __init__(self, policy, scheme, binary, proposal_limit=64, native_call_budget=10000):
        if type(proposal_limit) is not int or not 1 <= proposal_limit <= U32:
            raise ValueError('proposal limit must be a positive uint32')
        if type(native_call_budget) is not int or not 1 <= native_call_budget <= U32:
            raise ValueError('native call budget must be a positive uint32')
        self.native_call_budget = native_call_budget
        self.native_calls = 0
        self.policy, self.binary, self.proposal_limit = policy, binary, proposal_limit
        if tuple(scheme['n']) != tuple(policy.config.dimensions):
            raise ValueError('parent dimensions differ from policy configuration')
        verify_scheme(scheme)
        initial = operation(binary, scheme, 'inspect',
                            1 if policy.rng is None else policy.rng.state, policy.config.ceiling())
        if initial['outcome'] != 'inspected':
            raise ValueError('initial candidate capacity error')
        self.scheme, self.candidates = initial['scheme'], initial['candidates']
        if self.scheme['m'] != policy.rank:
            raise ValueError('policy/arithmetic initial rank mismatch')
        self.policy.bind_state(self.snapshot())
        self.trace = []

    def snapshot(self):
        return deepcopy({'scheme': self.scheme, 'candidates': self.candidates})

    def install_restart(self, scheme):
        """Verify and install the actual parent; preserve prior capture snapshots."""
        if scheme['n'] != self.scheme['n'] or scheme['z2'] != self.scheme['z2']:
            raise ValueError('restart must preserve dimensions and domain')
        verify_scheme(scheme)
        initial = operation(self.binary, scheme, 'inspect', self.policy.rng.state,
                            self.policy.config.ceiling())
        if initial['outcome'] != 'inspected':
            raise ValueError('restart candidate capacity error')
        state = {'scheme': initial['scheme'], 'candidates': initial['candidates']}
        controls = self.policy.controls
        self.policy.install_restart(scheme['m'], parent_verified=True, parent_state=state)
        if self.policy.controls != controls:
            self.scheme, self.candidates = deepcopy(initial['scheme']), deepcopy(initial['candidates'])
        return initial

    def take(self, kind):
        if self.snapshot() != self.policy.state:
            raise ValueError("policy and native ordered state diverged")
        rng = self.policy.rng.state
        operator = None
        if kind == 'expansion':
            # Policy has just consumed operator = next()%3, so current state
            # is that returned word. Proposal calls do not draw operator again.
            operator = rng % 3
            op = ('plus', 'random', 'existing')[operator]
            count = self.proposal_limit
        else:
            op = {'flip': 'flip', 'reduction': 'reduce'}[kind]
            count = 1
        if count > self.native_call_budget-self.native_calls:
            raise RuntimeError('reference native-call resource budget cannot cover invocation')
        result = operation(self.binary, self.scheme, op, rng,
                           self.policy.config.ceiling(), self.candidates, proposal_limit=count)
        observations = result['observations']
        self.native_calls += len(observations)
        self.trace.extend(deepcopy(observations))
        self.scheme, self.candidates = result['scheme'], result['candidates']
        last = result['outcome']
        if last == 'rank_blocked':
            raise AssertionError('controller called blocked arithmetic')
        return ArithmeticEvent(last, self.scheme['m'] if last == 'applied' else None,
                               result['draws'], operator, self.snapshot(),
                               result["removed_terms"] if kind in ("flip", "reduction") else None,
                               deepcopy(observations))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--compiler', default='clang++')
    args = parser.parse_args()
    print(build(args.build, args.compiler))
