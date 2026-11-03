import json, numpy as np
cg = {}
for l in open('cachegrind_scores.jsonl'):
    r = json.loads(l)
    if r.get('est_cycles') is not None:
        cg[(r['instance_id'], r['step'])] = r
rows = [json.loads(l) for l in open('swe_verified_durations.jsonl')]


def is_py(c):
    w = c.split()[0] if c.split() else ''
    return w in ('python', 'python3') or 'pytest' in c or w.startswith('PYTHONPATH')


sub = [r for r in rows if (r['instance_id'], r['step']) in cg and is_py(r['command'])]
print('n =', len(sub))
sub.sort(key=lambda r: r['duration_s'])

print('--- FASTEST 6 (by real duration) ---')
for r in sub[:6]:
    cyc = cg[(r['instance_id'], r['step'])]['est_cycles']
    print(f"  dur={r['duration_s']*1000:8.0f}ms  cyc={cyc:>14.0f}  {r['command'][:55]!r}")

print('--- SLOWEST 6 (by real duration) ---')
for r in sub[-6:]:
    cyc = cg[(r['instance_id'], r['step'])]['est_cycles']
    print(f"  dur={r['duration_s']*1000:8.0f}ms  cyc={cyc:>14.0f}  {r['command'][:55]!r}")

# correlation of raw numbers
y = np.array([r['duration_s'] for r in sub])
c = np.array([cg[(r['instance_id'], r['step'])]['est_cycles'] for r in sub])
print('\nduration range ms:', y.min()*1000, '-', y.max()*1000)
print('cycles range:', c.min(), '-', c.max())
print('corr(dur, cyc):', np.corrcoef(y, c)[0, 1])
