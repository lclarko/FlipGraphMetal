import argparse
from bisect import bisect_right
from collections import defaultdict
import json
from pathlib import Path
import re
import statistics
import xml.etree.ElementTree as ET


def select_samples(intervals, samples):
    starts = [span[0] for span in intervals]
    coverage = [0] * len(intervals)
    selected = []
    for timestamp, value in samples:
        index = bisect_right(starts, timestamp) - 1
        if index >= 0 and timestamp < intervals[index][1]:
            coverage[index] += 1
            selected.append(value)
    if not coverage or min(coverage) == 0:
        raise ValueError('counter samples do not cover every target compute segment')
    return selected, coverage


def check_dispatches(dispatches, focused):
    if focused and len(dispatches) != 3:
        raise ValueError('focused capture requires three distinct target command/encoder/submission groups')


def main():
    parser = argparse.ArgumentParser(description='Summarize device counters sampled during target compute intervals')
    parser.add_argument('toc', type=Path)
    parser.add_argument('data', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--capture-result', type=Path)
    args = parser.parse_args()
    toc = ET.parse(args.toc).getroot()
    runs = toc.findall('run')
    if len(runs) != 1 or runs[0].get('number') != '1':
        raise ValueError('expected one trace run numbered 1')
    run = runs[0]
    target = run.find('info/target/process')
    if target is None or target.get('return-exit-status') != '0':
        raise ValueError('trace target did not exit successfully')
    pid = target.get('pid')
    kernel = 'randomWalkKernel'
    if target.get('type') == 'attached':
        if not args.capture_result:
            raise ValueError('attached capture requires its completed target result')
        capture = json.loads(args.capture_result.read_text())
        kernel = capture.get('kernel', 'randomWalkKernel')
        if kernel not in ('randomWalkKernel', 'randomWalkCompactKernel'):
            raise ValueError('unknown captured kernel')
        if (str(capture.get('target_pid')) != pid or capture.get('exit_code') != 0
                or capture.get('recorder_exit') != 0 or capture.get('reason') is not None
                or capture.get('verification') != 'passed' or not capture.get('export_released')):
            raise ValueError('attached target did not complete verified capture and export')
    tables = {i: table.get('schema') for i, table in enumerate(run.findall('data/table'), 1)}
    columns = {
        'gpu-counter-info': ['timestamp', 'counter-id', 'name', 'max-value', 'accelerator-id', 'description',
                             'group-index', 'type', 'ring-buffer-count', 'require-weighted-accumulation', 'sample-interval'],
        'gpu-counter-value': ['timestamp', 'counter-id', 'value', 'accelerator-id', 'sample-index', 'ring-buffer-index'],
        'metal-gpu-intervals': ['start', 'duration', 'channel-name', 'frame-number', 'start-latency', 'event-depth',
                                'event-label', 'state', 'connection-UUID', 'color', 'process', 'gpu',
                                'channel-subtitle', 'iosurface-accesses', 'bytes', 'cmdbuffer-id', 'encoder-id', 'gpu-submission-id'],
    }
    refs = {}
    info = {}
    values = defaultdict(list)
    intervals = []
    dispatches = defaultdict(list)
    other_compute = []
    devices = set()
    node = None
    schema = None
    for event, element in ET.iterparse(args.data, events=('start', 'end')):
        if event == 'start':
            if element.tag == 'node':
                node = element
                match = re.search(r'/table\[(\d+)\]$', element.attrib['xpath'])
                if not match:
                    raise ValueError('unexpected exported table path')
                schema = tables[int(match[1])]
            continue
        if element.tag == 'schema':
            if element.get('name') != schema or [col.findtext('mnemonic') for col in element.findall('col')] != columns[schema]:
                raise ValueError('unexpected exported column order')
        if element.tag != 'row':
            continue
        for cell in element.iter():
            if cell.get('id'):
                refs[cell.get('id')] = (cell.text, cell.get('fmt'))
        cells = [refs[cell.get('ref')] if cell.get('ref') else (cell.text, cell.get('fmt')) for cell in element]
        data = [cell[0] for cell in cells]
        if schema == 'gpu-counter-info' and len(data) == 11:
            _, counter, name, maximum, accelerator, description, _, kind, _, weighted, interval = data
            info[(int(accelerator), int(counter))] = dict(name=name, maximum=maximum, description=description,
                                                       kind=kind, weighted=weighted, sample_interval=interval)
        elif schema == 'gpu-counter-value' and len(data) == 6:
            timestamp, counter, value, accelerator, _, _ = data
            values[(int(accelerator), int(counter))].append((int(timestamp), float(value)))
        elif schema == 'metal-gpu-intervals' and len(data) == 18:
            if data[2] == 'Compute' and data[7] == 'Active':
                span = (int(data[0]), int(data[0]) + int(data[1]))
                if span[1] <= span[0]:
                    raise ValueError('expected positive compute interval duration')
                if cells[10][1] == f'{target.get("name")} ({pid})':
                    if not (cells[6][1] or '').startswith(kernel + ':'):
                        raise ValueError('unexpected target compute encoder in flip capture')
                    identity = tuple(data[15:18])
                    if any(value in (None, '0') for value in identity):
                        raise ValueError('target compute segment lacks command identity')
                    dispatches[identity].append(span)
                    intervals.append(span)
                    devices.add(data[11])
                else:
                    other_compute.append(span)
        else:
            raise ValueError(f'unexpected schema or row width: {schema}, {len(data)}')
        node.remove(element)
        element.clear()
    if not intervals or len(devices) != 1 or len({key[0] for key in values}) != 1:
        raise ValueError('expected target compute intervals and counters from one GPU')
    check_dispatches(dispatches, target.get('type') == 'attached')
    intervals.sort()
    if any(a[1] > b[0] for a, b in zip(intervals, intervals[1:])):
        raise ValueError('overlapping target intervals require separate analysis')
    counters = {}
    for key, samples in values.items():
        selected, coverage = select_samples(intervals, samples)
        metadata = info[key]
        counters[metadata['name']] = dict(metadata, samples=len(selected), samples_per_segment=coverage, median=statistics.median(selected),
                                         minimum=min(selected), maximum_observed=max(selected))
    overlaps = [(max(a, c), min(b, d)) for a, b in intervals for c, d in other_compute if max(a, c) < min(b, d)]
    result = dict(target=dict(target.attrib), intervals_ns=intervals, active_ms=sum(b-a for a, b in intervals)/1e6,
                  other_compute_overlaps_ns=overlaps, counters=counters, coverage='complete for all target segments',
                  dispatches=[dict(command_buffer=key[0], encoder=key[1], submission=key[2], intervals_ns=spans)
                              for key, spans in dispatches.items()],
                  statistic='Unweighted sample statistics during target compute intervals. Device-wide counters include concurrent GPU work; these are not per-core measurements or time-weighted averages.')
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print('Target intervals:', len(intervals), 'active ms:', result['active_ms'])
    print('Other compute overlaps:', len(overlaps))
    for name, counter in counters.items():
        if 'Occupancy' in name or 'Limiter' in name:
            print(name, 'median:', counter['median'], 'maximum:', counter['maximum_observed'], 'samples:', counter['samples'])


if __name__ == '__main__':
    main()
