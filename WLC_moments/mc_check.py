import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import quad
from wlc_fig1 import run
rng = np.random.default_rng(7)

def mc(xi_over_L, nsamp=300000, nstep=3000):
    ds = 1.0/nstep; var = ds/xi_over_L
    u = np.zeros((nsamp,3)); u[:,2] = 1.0
    R = np.zeros((nsamp,3))
    for _ in range(nstep):
        g = rng.normal(size=(nsamp,3))
        g -= (g*u).sum(1, keepdims=True)*u          # project to tangent plane
        u = u + np.sqrt(var)*g
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        R += u*ds
    return np.linalg.norm(R, axis=1)

fig, ax = plt.subplots(figsize=(8,5.4))
cols = plt.cm.viridis(np.linspace(0,.9,8))
r = np.linspace(1e-6, 1-1e-9, 4000)
labels = {0.0025:'1/400',0.01:'1/100',1/30:'1/30',0.1:'1/10',0.2:'1/5',0.5:'1/2',1:'1',2:'2'}
for c,(xi,lab) in zip(cols, labels.items()):
    _,_,p = run(xi); k,beta,m = p
    y = r**(k+2)*np.maximum(1-r**beta,0)**m
    norm = quad(lambda t: t**(k+2)*max(1-t**beta,0)**m, 0,1, limit=400)[0]
    ax.plot(r, y/norm, lw=1.7, color=c, label=rf'$\xi/L={lab}$')
for c,xi in zip([cols[3],cols[6],cols[7]],[0.1,1.0,2.0]):
    s = mc(xi, nsamp=120000, nstep=2000)
    h,e = np.histogram(s, bins=90, range=(0,1), density=True)
    ax.plot(.5*(e[1:]+e[:-1]), h, 'o', ms=3.2, color=c, mfc='none')
ax.set_xlabel('$r=R/L$'); ax.set_ylabel('$P_L(r)$')
ax.set_title('Fig. 1 reproduced: exact-moment fits (lines) vs. own Monte Carlo (circles)')
ax.set_xlim(0,1); ax.legend(fontsize=8, ncol=2); fig.tight_layout()
fig.savefig('fig1_reproduction.png', dpi=150)
print('done')
