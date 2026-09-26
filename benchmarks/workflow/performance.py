"""Evaluate retained paired measurements against prospectively approved budgets.

No measurements are launched. Schema version 1 deliberately accepts no exclusions.
The field constants describe the required schema. Receipts hash the entire input,
including incomplete observations. Approval references are declarations to audit,
not signatures or proof of authorization.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def exact(obj, keys, label):
    if not isinstance(obj, dict) or set(obj) != set(keys.split()):
        raise ValueError(f'{label}: expected fields {keys}')


def number(value, label, positive=False):
    if type(value) not in (int, float) or not math.isfinite(value) or (positive and value <= 0):
        raise ValueError(f'{label}: invalid finite number')
    return value


def text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{label}: expected nonempty string')


def integer(value, label, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f'{label}: invalid integer')


def _fraction(a, b, x):
    # Modified Lentz continued fraction for the incomplete beta function.
    floor = 1e-300
    qab, qap, qam = a + b, a + 1, a - 1
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1 / (d if abs(d) > floor else floor)
    h = d
    for m in range(1, 10001):
        for aa in (m * (b-m) * x / ((qam+2*m)*(a+2*m)),
                   -(a+m)*(qab+m)*x / ((a+2*m)*(qap+2*m))):
            d = 1 + aa*d
            c = 1 + aa/c
            d = 1 / (d if abs(d) > floor else floor)
            c = c if abs(c) > floor else floor
            delta = d*c
            h *= delta
        if abs(delta-1) < 3e-14:
            return h
    raise ValueError('incomplete beta did not converge')


def beta(a, b, x):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    front = math.exp(math.lgamma(a+b)-math.lgamma(a)-math.lgamma(b)
                     + a*math.log(x)+b*math.log1p(-x))
    if x < (a+1)/(a+b+2):
        return front*_fraction(a, b, x)/a
    return 1-front*_fraction(b, a, 1-x)/b


def t_survival(t, df):
    if t < 0:
        return 1-t_survival(-t, df)
    return .5*beta(df/2, .5, df/(df+t*t))


def t_critical(tail, df):
    if not 0 < tail < .5 or df < 1:
        raise ValueError('invalid Student t parameters')
    lo, hi = 0.0, 1.0
    while t_survival(hi, df) > tail:
        hi *= 2
        if hi > 1e100:
            raise ValueError('unresolvable Student t quantile')
    for _ in range(180):
        mid = (lo+hi)/2
        if t_survival(mid, df) > tail:
            lo = mid
        else:
            hi = mid
    return (lo+hi)/2


# Paired differences: https://www.itl.nist.gov/div898/handbook/prc/section3/prc311.htm
# Family error allocation: https://www.itl.nist.gov/div898/handbook/prc/section4/prc463.htm
def interval(values, alpha):
    if len(values) < 2:
        raise ValueError('at least two pairs required')
    center = statistics.mean(values)
    radius = t_critical(alpha/2, len(values)-1)*statistics.stdev(values)/math.sqrt(len(values))
    return center-radius, center+radius


BUDGET_KEYS = 'version workload endpoint units direction estimand transform margin status approval rationale baseline_source candidate_source baseline_build candidate_build absolute_limit'
PROTOCOL_KEYS = 'version alpha mandatory_endpoints prospective_pairs precision_criterion looks assumptions protocol_id matched_identity'
ENDPOINT_KEYS = 'workload endpoint units direction estimand transform'
MATCH_KEYS = 'fixture config seed work build_settings shader_mode hardware'
ROW_KEYS = 'pair_id order complete baseline candidate baseline_source candidate_source baseline_build candidate_build baseline_identity candidate_identity resource_violation error'


def evaluate(document):
    exact(document, 'version budget budget_sha256 protocol protocol_sha256 observations look previous precision_assessment', 'document')
    if document['version'] != 1:
        raise ValueError('unsupported document version')
    budget, protocol = document['budget'], document['protocol']
    exact(budget, BUDGET_KEYS, 'budget')
    exact(protocol, PROTOCOL_KEYS, 'protocol')
    if document['budget_sha256'] != digest(budget) or document['protocol_sha256'] != digest(protocol):
        raise ValueError('budget or protocol content hash mismatch')
    for field in ('workload', 'endpoint', 'units', 'estimand', 'rationale', 'baseline_source', 'candidate_source', 'baseline_build', 'candidate_build'):
        text(budget[field], field)
    integer(budget['version'], 'budget version')
    if budget['direction'] not in ('lower', 'higher') or budget['transform'] not in ('log-ratio', 'difference'):
        raise ValueError('invalid direction or transform')
    if budget['status'] not in ('approved', 'unapproved'):
        raise ValueError('invalid approval status')
    # An unapproved budget may explicitly leave its margin unset.
    if budget['margin'] is not None or budget['status'] == 'approved':
        number(budget['margin'], 'margin')
        if budget['margin'] < 0 or (budget['transform'] == 'log-ratio' and budget['direction'] == 'higher' and budget['margin'] >= 1):
            raise ValueError('invalid practical margin')
    if budget['status'] == 'approved':
        exact(budget['approval'], 'reference version', 'approval')
        text(budget['approval']['reference'], 'approval reference')
        text(budget['approval']['version'], 'approval version')
    elif budget['approval'] is not None:
        raise ValueError('unapproved budget must not claim approval')
    if budget['absolute_limit'] is not None:
        number(budget['absolute_limit'], 'absolute limit')
    if protocol['version'] != 1 or protocol['looks'] != [6, 12]:
        raise ValueError('protocol requires version 1 and looks [6,12]')
    number(protocol['alpha'], 'alpha', True)
    if not 0 < protocol['alpha'] < 1:
        raise ValueError('alpha outside (0,1)')
    endpoints = protocol['mandatory_endpoints']
    if not isinstance(endpoints, list) or not endpoints:
        raise ValueError('mandatory endpoint roster required')
    endpoint_ids = set()
    for endpoint in endpoints:
        exact(endpoint, ENDPOINT_KEYS, 'mandatory endpoint')
        for key, value in endpoint.items():
            text(value, key)
        if endpoint['direction'] not in ('lower', 'higher') or endpoint['transform'] not in ('log-ratio', 'difference'):
            raise ValueError('invalid mandatory endpoint metric')
        key = (endpoint['workload'], endpoint['endpoint'])
        if key in endpoint_ids:
            raise ValueError('duplicate mandatory endpoint')
        endpoint_ids.add(key)
    if {key: budget[key] for key in ENDPOINT_KEYS.split()} not in endpoints:
        raise ValueError('budget endpoint not bound by mandatory roster')
    family_size = len(endpoints)
    prospective = protocol['prospective_pairs']
    if not isinstance(prospective, list) or len(prospective) != 12:
        raise ValueError('exactly twelve prospective pairs required')
    prospective_ids = set()
    for pair in prospective:
        exact(pair, 'pair_id order identity', 'prospective pair')
        text(pair['pair_id'], 'prospective pair id')
        if pair['pair_id'] in prospective_ids or pair['order'] not in ('AB', 'BA'):
            raise ValueError('duplicate or invalid prospective pair')
        prospective_ids.add(pair['pair_id'])
        exact(pair['identity'], MATCH_KEYS, 'prospective identity')
        for key, value in pair['identity'].items():
            text(value, key)
    for block in (prospective[:6], prospective[6:]):
        if sum(pair['order'] == 'AB' for pair in block) != 3:
            raise ValueError('each prospective six-pair block must balance AB and BA')
    text(protocol['precision_criterion'], 'prospective precision criterion')
    precision = document['precision_assessment']
    exact(precision, 'adequate evidence', 'precision assessment')
    if type(precision['adequate']) is not bool:
        raise ValueError('precision adequacy must be explicitly assessed')
    text(precision['evidence'], 'precision evidence')
    text(protocol['protocol_id'], 'protocol id')
    if type(protocol['assumptions']) is not bool:
        raise ValueError('assumptions must be explicitly assessed')
    if protocol['matched_identity'] != MATCH_KEYS.split():
        raise ValueError('required matching identity fields absent or reordered')
    if type(document['look']) is not int or document['look'] not in (1, 2):
        raise ValueError('only two looks allowed')
    rows = document['observations']
    if not isinstance(rows, list) or len(rows) > protocol['looks'][document['look']-1]:
        raise ValueError('invalid observation count')
    if document['look'] == 1:
        if document['previous'] is not None:
            raise ValueError('first look cannot have a previous receipt')
    else:
        previous = document['previous']
        exact(previous, 'verdict budget_sha256 protocol_sha256 observations_sha256 precision_assessment', 'previous')
        if (previous['verdict'] != 'INCONCLUSIVE' or previous['budget_sha256'] != document['budget_sha256']
                or previous['protocol_sha256'] != document['protocol_sha256']
                or previous['observations_sha256'] != digest(rows[:6])):
            raise ValueError('second look requires unchanged inconclusive first six pairs')
        first = dict(document, look=1, previous=None, observations=rows[:6],
                     precision_assessment=previous['precision_assessment'])
        if evaluate(first)['verdict'] != 'INCONCLUSIVE':
            raise ValueError('first six pairs do not independently yield INCONCLUSIVE')
    seen, values, orders = set(), [], []
    incomplete = False
    violation = False
    for index, row in enumerate(rows):
        exact(row, ROW_KEYS, 'observation')
        text(row['pair_id'], 'pair id')
        if row['pair_id'] in seen:
            raise ValueError('duplicate pair id')
        seen.add(row['pair_id'])
        if row['order'] not in ('AB', 'BA'):
            raise ValueError('invalid pair order')
        orders.append(row['order'])
        planned = prospective[index]
        if row['pair_id'] != planned['pair_id'] or row['order'] != planned['order']:
            raise ValueError('observation differs from prospective order or pair roster')
        if type(row['complete']) is not bool or type(row['resource_violation']) is not bool:
            raise ValueError('invalid completion/resource status')
        for field in ('baseline_source', 'candidate_source', 'baseline_build', 'candidate_build'):
            if row[field] != budget[field]:
                raise ValueError('source or build identity mismatch')
        for field in ('baseline_identity', 'candidate_identity'):
            exact(row[field], MATCH_KEYS, 'matched identity')
            for key, value in row[field].items():
                text(value, key)
        if row['baseline_identity'] != planned['identity'] or row['candidate_identity'] != planned['identity']:
            raise ValueError('both observations must match the frozen prospective work identity')
        if row['baseline_identity'] != row['candidate_identity']:
            raise ValueError('matched work/configuration identity mismatch')
        violation |= row['resource_violation']
        if row['error'] is not None:
            text(row['error'], 'error')
        if not row['complete'] or row['error'] is not None:
            incomplete = True
        for field in ('baseline', 'candidate'):
            if row[field] is not None:
                number(row[field], field, budget['transform'] == 'log-ratio')
        if row['complete'] and (row['baseline'] is None or row['candidate'] is None):
            raise ValueError('complete observation missing measurement')
        c, b = row['candidate'], row['baseline']
        if c is not None and budget['absolute_limit'] is not None:
            violation |= c > budget['absolute_limit'] if budget['direction'] == 'lower' else c < budget['absolute_limit']
        if row['complete'] and row['error'] is None:
            value = math.log(c)-math.log(b) if budget['transform'] == 'log-ratio' else c-b
            values.append(value if budget['direction'] == 'lower' else -value)
    result = {'version': 1, 'input_sha256': digest(document), 'budget_sha256': digest(budget),
              'protocol_sha256': digest(protocol), 'observations_sha256': digest(rows),
              'look': document['look'], 'pairs_retained': len(rows), 'verdict': 'INCONCLUSIVE',
              'change': 'unresolved', 'interval': None, 'reason': '', 'resource_violation': violation,
              'budget_readiness': budget['status'] == 'approved', 'family_size': family_size,
              'precision_assessment': precision, 'precision_criterion': protocol['precision_criterion']}
    if violation:
        result.update(verdict='FAIL', reason='absolute resource budget violated')
    elif budget['status'] != 'approved':
        result['reason'] = 'practical budget unapproved'
    elif incomplete or len(rows) != protocol['looks'][document['look']-1]:
        result['reason'] = 'incomplete or insufficient retained observations'
    elif orders.count('AB') != orders.count('BA') or not protocol['assumptions']:
        result['reason'] = 'unbalanced order or unresolved statistical assumptions'
    elif not precision['adequate']:
        result['reason'] = 'prospective precision criterion not met'
    else:
        alpha = protocol['alpha']/(family_size*len(protocol['looks']))
        lower, upper = interval(values, alpha)
        margin = budget['margin']
        if budget['transform'] == 'log-ratio':
            margin = math.log1p(margin) if budget['direction'] == 'lower' else -math.log1p(-margin)
        verdict = 'PASS' if upper <= margin else 'FAIL' if lower > margin else 'INCONCLUSIVE'
        change = 'slowdown established' if lower > 0 else 'improvement established' if upper < 0 else 'unresolved'
        result.update(verdict=verdict, change=change, interval=[lower, upper], mean=statistics.mean(values),
                      approved_degradation=margin, endpoint_alpha=alpha,
                      reason=('PASS within the approved tolerance; a slowdown was established.'
                              if verdict == 'PASS' and lower > 0 else 'paired interval compared with approved budget'))
    result['receipt_sha256'] = digest(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    def reject_constant(value):
        raise ValueError('nonfinite JSON constant: '+value)
    def unique_fields(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('duplicate JSON field: '+key)
            result[key] = value
        return result
    document = json.loads(args.input.read_text(), parse_constant=reject_constant, object_pairs_hook=unique_fields)
    result = evaluate(document)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(result['verdict']+': '+result['reason'])
    raise SystemExit(0 if result['verdict'] == 'PASS' else 1 if result['verdict'] == 'FAIL' else 2)


if __name__ == '__main__':
    main()
