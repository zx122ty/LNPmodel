"""Final verdict: does the protocol flip the base ranking? (end-to-end, Sheet2)"""
import numpy as np
from scipy.stats import ttest_rel, wilcoxon

SEQ = {'stock':[0.5359,0.8097,0.7278,0.7396,0.6456],
       'v2'   :[0.5312,0.7992,0.7481,0.7212,0.5934],
       'v3'   :[0.5124,0.7642,0.7319,0.7337,0.5901],
       'v4'   :[0.5124,0.7780,0.7547,0.7242,0.6750]}
SCA = {'stock':[0.592,0.707,0.703,0.689,0.620],
       'v2'   :[0.556,0.680,0.689,0.720,0.612],
       'v3'   :[0.573,0.670,0.710,0.689,0.652],
       'v4'   :[0.586,0.667,0.644,0.641,0.640]}
RSP_SEQ = {'stock':0.8364,'v2':0.8340,'v3':0.8297,'v4':0.8391}

print('='*78); print('  端到端 fold-mean R²：两协议对照'); print('='*78)
print(f'  {"base":8} {"scaffold":>18} {"sequential":>18} {"lift":>9} {"relaxSp(seq)":>13}')
for b in SEQ:
    s,q = np.mean(SCA[b]), np.mean(SEQ[b])
    print(f'  {b:8} {s:>10.4f}±{np.std(SCA[b],ddof=1):.3f} {q:>10.4f}±{np.std(SEQ[b],ddof=1):.3f} {q-s:>+9.4f} {RSP_SEQ[b]:>13.4f}')

print('\n' + '='*78); print('  各底座 vs stock：逐折配对检验'); print('='*78)
for name, D in (('scaffold', SCA), ('sequential', SEQ)):
    print(f'  -- {name}')
    for b in ('v2','v3','v4'):
        d = np.array(D[b]) - np.array(D['stock'])
        tp = ttest_rel(D[b], D['stock'])[1]; wp = wilcoxon(D[b], D['stock'])[1]
        flag = '' if tp > 0.05 else '  <-- SIGNIFICANT'
        print(f'     {b}-stock = {d.mean():+.4f}  folds={np.round(d,3)}  t p={tp:.3f}  wil p={wp:.3f}{flag}')

print('\n' + '='*78); print('  ★ v4 的协议依赖性（关键对照）'); print('='*78)
ds = np.mean(SCA['v4'])-np.mean(SCA['stock']); dq = np.mean(SEQ['v4'])-np.mean(SEQ['stock'])
print(f'  v4-stock  scaffold   = {ds:+.4f}   (排名 4/4，垫底)')
print(f'  v4-stock  sequential = {dq:+.4f}   (排名 2/4，仅次于 stock)')
print(f'  协议造成的相对位移 = {dq-ds:+.4f}')
print(f'  v4 自身抬升 = {np.mean(SEQ["v4"])-np.mean(SCA["v4"]):+.4f}'
      f'  (最大；stock 仅 {np.mean(SEQ["stock"])-np.mean(SCA["stock"]):+.4f})')
print(f'  relaxed-Spearman 下 v4 = {RSP_SEQ["v4"]:.4f} 为四底座最高')

print('\n' + '='*78); print('  排序对比'); print('='*78)
for name, D in (('scaffold', SCA), ('sequential', SEQ)):
    order = sorted(D, key=lambda b: -np.mean(D[b]))
    print(f'  {name:11} (R²)      : ' + ' > '.join(order))
print(f'  {"sequential":11} (relaxSp) : ' + ' > '.join(sorted(RSP_SEQ, key=lambda b: -RSP_SEQ[b])))

print('\n' + '='*78); print('  与 ECFP4 基线对照'); print('='*78)
print(f'  {"":28} {"scaffold":>10} {"sequential":>12}')
print(f'  {"ECFP4+desc RF (frozen)":28} {0.5532:>10.4f} {0.6903:>12.4f}')
print(f'  {"best end-to-end neural":28} {max(np.mean(v) for v in SCA.values()):>10.4f} '
      f'{max(np.mean(v) for v in SEQ.values()):>12.4f}')
