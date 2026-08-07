# Reconstruction of `pgm15` — Hamprecht & Kleinert, cond-mat/0305226

The Mathematica notebook referenced as [7] in the arXiv preprint
(`http://www.physik.fu-berlin.de/~kleinert/b5/pgm15`) is dead — Kleinert's
FU-Berlin personal pages were removed. These scripts re-implement it.

## Files

| file | what it does |
|---|---|
| `wlc_moments.py` | Exact symbolic Rayleigh–Schrödinger recursion (Sec. II). Gives `<R^2n>(L)` as closed-form rational combinations of `L^j exp(-l(l+D-2)L/2)`, for arbitrary `D`. |
| `check_independent.py` | Independent validation: numerical `<0|exp(-HL)|0>` by matrix exponential in the Legendre basis. No recursion used. |
| `wlc_fig1.py` | Fast mpmath version of the same recursion + 3-parameter fit of Sec. III. |
| `mc_check.py` | Own Monte Carlo of the wormlike chain + final figure. |
| `fig1_reproduction.png` | Reproduction of Fig. 1. |

## Units

`kappa_bar = 1`, so `xi = 2/(D-1)` (= 1 in D=3) and `L` is free. The paper's
`xi/L` at `L=1` corresponds to `L = 1/(xi/L)` here.

## Validation

* `<R^2> = 2{xi L - xi^2 [1 - e^{-L/xi}]}` — reproduced exactly.
* `<R^4>` — reproduced exactly, **except** that the arXiv preprint's last term
  reads `2(D-5)^5/(D^3(D+1)^2)`. That is a typo. The correct coefficient,
  obtained here from `D = 3,4,5,6` and confirmed against the published
  version (Phys. Rev. E 71, 031803 (2005)), is

      + (D-1)^5 / (D^3 (D+1)^2) * exp(-2DL/((D-1)xi))     [inside the 4 xi^4 bracket]

* Fitted `(k, beta, m)` reproduce the asymptotics quoted in the published paper:
  `k -> -xi`, `beta -> 2 + 2xi`, `m -> 3/(4xi)` as `xi -> 0`;
  `k -> 10xi - 7/2`, `beta -> 40xi + 5`, `m -> 10` at large `xi`.
* Own Monte Carlo agrees with the fitted curves.

## Fitted parameters (D = 3, L = 1)

| xi/L | moments matched | k | beta | m |
|---|---|---|---|---|
| 1/400 | 2,4,6   | -0.0030 | 2.0040 | 300.34 |
| 1/100 | 2,4,6   | -0.0184 | 2.0234 | 75.18 |
| 1/30  | 2,4,6   | -0.1208 | 2.1757 | 23.14 |
| 1/10  | 2,4,6   | -0.4859 | 3.6079 | 9.974 |
| 1/5   | 2,4,6   | -0.3144 | 8.1701 | 9.653 |
| 1/2   | 4,6,8   | 2.0218  | 26.161 | 15.13 |
| 1     | 8,10,12 | 6.7685  | 47.164 | 12.50 |
| 2     | 16,18,20| 16.499  | 85.499 | 10.46 |
