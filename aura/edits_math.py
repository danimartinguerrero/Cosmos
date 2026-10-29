"""
Edits de formas diferenciales y teoremas fundamentales del cálculo vectorial
(basados en «El teorema de Stokes generalizado mediante formas diferenciales»).

Estilo: sin música (pista muda con rejilla de pulsos a 150 bpm), cortes cada 1–3 s,
solo fórmulas —sin texto explicativo— sobre interpretaciones geométricas animadas.
"""
from .edit import Edit, Shot, Fx
from .scenes_math import (FTC, GradientPaths, FieldFlow, GreenTiling, Parametrization, Charts, StokesNet,
                          GaussSphere, OneForm, Wedge, BoundaryCube, Pullback, EllipseArea, FormulaVoid)

BPM = 150
TOP, MID, LOW = 300, 960, 1610

STOKES = r'$\int_M d\omega=\int_{\partial M}\omega$'
FTC_TEX = r'$\int_a^b f\,^\prime(x)\,dx=f(b)-f(a)$'
GRAD = r'$\int_\gamma \nabla f\cdot d\mathbf{r}=f(q)-f(p)$'
GREEN = r'$\iint_D\left(\frac{\partial Q}{\partial x}-\frac{\partial P}{\partial y}\right)dx\,dy=\oint_{\partial D}P\,dx+Q\,dy$'
STOKES3 = r'$\iint_S(\nabla\times\mathbf{F})\cdot d\mathbf{S}=\oint_{\partial S}\mathbf{F}\cdot d\mathbf{r}$'
GAUSS = r'$\iiint_V\nabla\cdot\mathbf{F}\,dV=\oiint_{\partial V}\mathbf{F}\cdot d\mathbf{S}$'


def _edit(name, shots, formulas, beats_total=None):
    return Edit(name=name, grade='lab', beats=True, music=('silence', dict(bpm=BPM, hit_every=8)),
                beat_zoom=0.03, hit_flash=0.45, fade_in=0.08, tail=1.2, shots=shots, formulas=formulas)


def _cum(durs):
    t = [0]
    for d in durs:
        t.append(t[-1] + d)
    return t


def stokes_generalizado():
    """M1 · Cinco teoremas, un solo enunciado."""
    shots = [
        Shot(FormulaVoid(seed=1), 4, trans='cut'),
        Shot(FTC(), 6, trans='flash', tdur=0.3, speed=1.6),
        Shot(GradientPaths(), 6, trans='zoom', tdur=0.3, speed=1.6),
        Shot(FieldFlow(seed=2, kind='curl', overlay='loop', loop_r=1.5), 4, trans='whip', tdur=0.3, offset=1.0),
        Shot(GreenTiling(levels=(3, 6, 12, 24)), 8, trans='flash', tdur=0.3),
        Shot(Parametrization(kind='dome', morph=(0.05, 0.7), orbit=0.8), 4, trans='glitch', tdur=0.3),
        Shot(StokesNet(freq=3.2), 8, trans='zoom', tdur=0.3),
        Shot(FieldFlow(seed=3, kind='source', overlay='box', color='#ffb35c'), 4, trans='whip', tdur=0.3, offset=1.0),
        Shot(GaussSphere(seed=4), 8, trans='flash', tdur=0.3, speed=1.4),
        Shot(BoundaryCube(), 6, trans='glitch', tdur=0.3, speed=1.4),
        Shot(FormulaVoid(seed=5), 12, trans='flash', tdur=0.3),
        Shot(Charts(kind='sphere'), 6, trans='zoom', tdur=0.3, speed=1.6),
    ]
    t = _cum([s.dur for s in shots])
    fx = [
        Fx(t[0] + 0.3, t[1], STOKES, (540, MID), 104, 'slam'),
        Fx(t[1] + 0.4, t[2], FTC_TEX, (540, TOP), 72, 'pop'),
        Fx(t[2] + 0.4, t[3], GRAD, (540, LOW + 60), 72, 'glitch'),
        Fx(t[3] + 0.3, t[4], r'$\oint_C\mathbf{F}\cdot d\mathbf{r}$', (540, TOP), 84, 'pop'),
        Fx(t[4] + 0.4, t[5], GREEN, (540, TOP - 20), 64, 'wipe'),
        Fx(t[5] + 0.3, t[6], r'$\varphi:D\subset\mathbb{R}^2\longrightarrow M$', (540, TOP), 80, 'rise'),
        Fx(t[6] + 0.4, t[7], STOKES3, (540, LOW + 40), 66, 'wipe', glow='#ff8fc4'),
        Fx(t[7] + 0.3, t[8], r'$\nabla\cdot\mathbf{F}$', (540, TOP), 100, 'slam', glow='#ffb35c'),
        Fx(t[8] + 0.4, t[9], GAUSS, (540, LOW + 60), 70, 'glitch', glow='#ffb35c'),
        Fx(t[9] + 0.3, t[10] - 3, r'$\partial\partial M=\varnothing$', (540, TOP), 96, 'slam'),
        Fx(t[9] + 3.0, t[10], r'$d^{\,2}=0$', (540, LOW), 110, 'slam', glow='#ffc56b'),
        # cascada final: los cinco teoremas y su síntesis
        Fx(t[10] + 0.2, t[10] + 7.4, FTC_TEX, (540, 360), 54, 'rise'),
        Fx(t[10] + 1.4, t[10] + 7.4, GRAD, (540, 600), 54, 'rise'),
        Fx(t[10] + 2.6, t[10] + 7.4, GREEN, (540, 830), 50, 'rise'),
        Fx(t[10] + 3.8, t[10] + 7.4, STOKES3, (540, 1060), 54, 'rise'),
        Fx(t[10] + 5.0, t[10] + 7.4, GAUSS, (540, 1280), 54, 'rise'),
        Fx(t[10] + 7.7, t[11], STOKES, (540, MID), 120, 'slam', glow='#ffc56b'),
        Fx(t[11] + 0.2, t[12] + 1, STOKES, (540, LOW + 40), 92, 'blur', glow='#ffc56b'),
    ]
    return _edit('M1_stokes_generalizado', shots, fx)


def formas_diferenciales():
    """M2 · 1-formas, producto exterior, diferencial exterior, pullback."""
    shots = [
        Shot(FormulaVoid(seed=7), 4, trans='cut'),
        Shot(OneForm(kind='constant'), 4, trans='flash', tdur=0.3),
        Shot(OneForm(kind='curved'), 4, trans='whip', tdur=0.3),
        Shot(Wedge(), 6, trans='zoom', tdur=0.3, speed=1.5),
        Shot(Wedge(swap=True), 4, trans='glitch', tdur=0.3, offset=2.0, speed=1.5),
        Shot(Pullback(), 6, trans='flash', tdur=0.3, speed=1.5),
        Shot(Parametrization(kind='torus', morph=(0.05, 0.65), orbit=0.9), 6, trans='zoom', tdur=0.3),
        Shot(FieldFlow(seed=8, kind='curl', overlay='paddle', color='#ff7ab8'), 4, trans='whip', tdur=0.3, offset=1.0),
        Shot(FieldFlow(seed=9, kind='source', color='#ffb35c'), 4, trans='flash', tdur=0.3, offset=1.0),
        Shot(GradientPaths(), 4, trans='glitch', tdur=0.3, offset=3.0, speed=1.6),
        Shot(BoundaryCube(), 6, trans='zoom', tdur=0.3, speed=1.5),
        Shot(Charts(kind='torus'), 6, trans='flash', tdur=0.3, speed=1.5),
        Shot(Parametrization(kind='sphere', morph=(0.0, 0.5), orbit=1.0, cell=(0.45, 0.2)), 6, trans='zoom', tdur=0.3),
    ]
    t = _cum([s.dur for s in shots])
    fx = [
        Fx(t[0] + 0.3, t[1], r'$d:\Omega^k(U)\longrightarrow\Omega^{k+1}(U)$', (540, MID), 84, 'slam'),
        Fx(t[1] + 0.3, t[2], r'$\omega_x(v)=\sum_i a_i(x)\,v_i=\mathbf{a}(x)\cdot v$', (540, LOW), 64, 'pop'),
        Fx(t[2] + 0.3, t[3], r'$df=\sum_i\frac{\partial f}{\partial x_i}\,dx_i$', (540, LOW), 76, 'glitch'),
        Fx(t[3] + 0.3, t[4], r'$(dx_i\wedge dx_j)(u,v)=u_i v_j-u_j v_i$', (540, TOP), 66, 'wipe'),
        Fx(t[4] + 0.3, t[5], r'$dx_i\wedge dx_j=-dx_j\wedge dx_i$', (540, TOP), 74, 'slam'),
        Fx(t[5] + 0.3, t[6], r'$dx\wedge dy=\det\frac{\partial(x,y)}{\partial(u,v)}\,du\wedge dv$', (540, TOP), 66, 'wipe', glow='#ffc56b'),
        Fx(t[6] + 0.3, t[7], r'$\varphi^*\omega=g(u)\,du_1\wedge\cdots\wedge du_k$', (540, LOW), 70, 'rise'),
        Fx(t[7] + 0.3, t[8], r'$d\,\omega_\mathbf{F}=\Phi_{\nabla\times\mathbf{F}}$', (540, TOP), 88, 'slam', glow='#ff8fc4'),
        Fx(t[8] + 0.3, t[9], r'$d\,\Phi_\mathbf{F}=(\nabla\cdot\mathbf{F})\,dx\wedge dy\wedge dz$', (540, TOP), 66, 'glitch', glow='#ffb35c'),
        Fx(t[9] + 0.3, t[10], r'$df=\omega_{\nabla f}$', (540, LOW + 60), 96, 'slam'),
        Fx(t[10] + 0.3, t[10] + 2.4, r'$d(d\omega)=0$', (540, TOP), 92, 'slam'),
        Fx(t[10] + 2.6, t[11], r'$\nabla\times(\nabla f)=0\qquad\nabla\cdot(\nabla\times\mathbf{F})=0$', (540, LOW + 40), 60, 'wipe'),
        Fx(t[11] + 0.3, t[12], r'$\int_M\omega=\int_D\varphi^*\omega$', (540, LOW), 84, 'blur'),
        Fx(t[12] + 0.3, t[13] + 1, STOKES, (540, TOP), 104, 'slam', glow='#ffc56b'),
    ]
    return _edit('M2_formas_diferenciales', shots, fx)


def geometria_teoremas():
    """M3 · Interpretación geométrica: rotacional, divergencia, cancelación, ejemplos y Maxwell."""
    shots = [
        Shot(FieldFlow(seed=11, kind='curl', overlay='paddle', color='#ff7ab8'), 6, trans='cut', offset=1.0),
        Shot(FieldFlow(seed=12, kind='source', overlay='box', color='#ffb35c'), 6, trans='flash', tdur=0.3, offset=1.0),
        Shot(FieldFlow(seed=13, kind='swirl', overlay='loop', loop_r=0.9, loop_c=(0.6, 0.8)), 4, trans='whip', tdur=0.3, offset=1.0),
        Shot(GreenTiling(levels=(4, 8, 16, 32)), 8, trans='zoom', tdur=0.3),
        Shot(EllipseArea(), 6, trans='flash', tdur=0.3, speed=1.5),
        Shot(GaussSphere(seed=14), 6, trans='glitch', tdur=0.3, speed=1.5),
        Shot(FieldFlow(seed=15, kind='dipole', overlay='charges', color='#9fd8ff', vmax=3.0, n=2200, trail=14), 6, trans='zoom', tdur=0.3, offset=1.0),
        Shot(FieldFlow(seed=16, kind='curl', overlay='loop', loop_r=1.2, color='#7cffc4'), 6, trans='whip', tdur=0.3, offset=1.0),
        Shot(Parametrization(kind='wave', morph=(0.0, 0.5), orbit=0.9), 4, trans='flash', tdur=0.3),
        Shot(StokesNet(freq=3.0), 8, trans='zoom', tdur=0.3),
    ]
    t = _cum([s.dur for s in shots])
    fx = [
        Fx(t[0] + 0.3, t[1], r'$\nabla\times\mathbf{F}$', (540, TOP), 110, 'slam', glow='#ff8fc4'),
        Fx(t[1] + 0.3, t[2], r'$\nabla\cdot\mathbf{F}=\lim_{V\to 0}\frac{1}{V}\oiint_{\partial V}\mathbf{F}\cdot d\mathbf{S}$', (540, TOP), 66, 'wipe', glow='#ffb35c'),
        Fx(t[2] + 0.3, t[3], r'$(\nabla\times\mathbf{F})\cdot\mathbf{n}=\lim_{A\to 0}\frac{1}{A}\oint_{\partial A}\mathbf{F}\cdot d\mathbf{r}$', (540, LOW + 40), 62, 'glitch'),
        Fx(t[3] + 0.3, t[4], r'$\sum_i\oint_{\partial R_i}\omega=\oint_{\partial D}\omega$', (540, TOP), 80, 'pop', glow='#ffc56b'),
        Fx(t[4] + 0.3, t[4] + 2.6, r'$\omega=\frac{1}{2}(x\,dy-y\,dx)$', (540, TOP), 80, 'slam'),
        Fx(t[4] + 2.8, t[5], r'$\frac{1}{2}\oint_C(x\,dy-y\,dx)=\pi ab$', (540, LOW), 74, 'wipe', glow='#ffc56b'),
        Fx(t[5] + 0.3, t[6], r'$\oiint_{S^2}\mathbf{r}\cdot d\mathbf{S}=\iiint_B 3\,dV=4\pi$', (540, LOW + 60), 68, 'glitch', glow='#ffb35c'),
        Fx(t[6] + 0.3, t[7], r'$\nabla\cdot\mathbf{E}=\frac{\rho}{\varepsilon_0}$', (540, LOW + 40), 104, 'slam'),
        Fx(t[7] + 0.3, t[8], r'$\oint_{\partial S}\mathbf{E}\cdot d\mathbf{r}=-\frac{d}{dt}\iint_S\mathbf{B}\cdot d\mathbf{S}$', (540, TOP), 64, 'wipe', glow='#7cffc4'),
        Fx(t[8] + 0.3, t[9], r'$F=dA\quad\Rightarrow\quad dF=0$', (540, TOP), 92, 'slam'),
        Fx(t[9] + 0.3, t[10] + 1, STOKES, (540, LOW + 40), 108, 'slam', glow='#ffc56b'),
    ]
    return _edit('M3_geometria_de_los_teoremas', shots, fx)


EDITS = {
    'M1_stokes_generalizado': stokes_generalizado,
    'M2_formas_diferenciales': formas_diferenciales,
    'M3_geometria_de_los_teoremas': geometria_teoremas,
}
