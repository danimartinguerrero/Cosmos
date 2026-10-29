"""
Escenas de geometría diferencial y cálculo vectorial: teorema fundamental del cálculo,
gradiente e independencia del camino, campos vectoriales (rotacional, divergencia),
Green por cancelación interior, Stokes sobre superficies con el mismo borde, Gauss,
1-formas como pilas de hiperplanos, producto exterior y áreas orientadas,
∂∂M = ∅, pullback / jacobiano, parametrizaciones y cartas sobre variedades.

Todo es dibujo lineal luminoso sobre negro; las fórmulas se superponen desde edit.py.
"""
import math
import numpy as np
from numba import njit, prange

from .core import (W, H, FPS, canvas, resize, blur, splat, splat_fast, fbm3, add_glow_point, hexcol,
                   ease_in_out, ease_out, ease_in, look_at, project, rot_x, rot_y, rot_z, line, polyline,
                   polyline_fade, disc, smoothstep)
from .scene import Scene


def C(h, g=1.0):
    return hexcol(h, g)


CYAN = C('#53e0ff'); GOLD = C('#ffc56b'); MAG = C('#ff5fa2'); VIO = C('#9b7bff'); WHITE = C('#f4f7ff')
BLUE = C('#3f7dff'); ORANGE = C('#ff8a3c'); GREEN = C('#5dffb0')


# ───────────────────────────── utilidades de dibujo ─────────────────────────────

class Plane:
    """Sistema de coordenadas 2D -> píxeles."""

    def __init__(self, scale=210.0, cx=W / 2, cy=H / 2):
        self.s = scale; self.cx = cx; self.cy = cy

    def __call__(self, x, y):
        return self.cx + np.asarray(x) * self.s, self.cy - np.asarray(y) * self.s


def pl(img, X, Y, col, width=2.0, glow=4.0, alpha=None):
    pts = np.stack([np.asarray(X, np.float32), np.asarray(Y, np.float32)], 1).astype(np.float32)
    if len(pts) < 2:
        return
    if alpha is None:
        polyline(img, pts, float(col[0]), float(col[1]), float(col[2]), width, glow)
    else:
        polyline_fade(img, pts, np.asarray(alpha, np.float32), float(col[0]), float(col[1]), float(col[2]), width, glow)


def arrow(img, x0, y0, x1, y1, col, width=2.0, glow=3.0, head=14.0):
    line(img, x0, y0, x1, y1, float(col[0]), float(col[1]), float(col[2]), width, glow)
    dx, dy = x1 - x0, y1 - y0
    L = math.hypot(dx, dy)
    if L < 1e-6:
        return
    ux, uy = dx / L, dy / L
    h = min(head, L * 0.45)
    for s in (1, -1):
        ax = x1 - ux * h + s * (-uy) * h * 0.5
        ay = y1 - uy * h + s * ux * h * 0.5
        line(img, x1, y1, ax, ay, float(col[0]), float(col[1]), float(col[2]), width, glow)


def pl3(img, P, cam, col, fov=45, width=1.6, glow=3.0, depth_fade=None, alpha=1.0):
    x, y, z = project(np.asarray(P, np.float64), cam, fov)
    pts = np.stack([x, y], 1).astype(np.float32)
    ok = z > 0.05
    al = np.full(len(pts) - 1, alpha, np.float32)
    if depth_fade is not None:
        zn = (z[:-1] - depth_fade[0]) / (depth_fade[1] - depth_fade[0])
        al = al * np.clip(1.15 - 0.85 * zn, 0.12, 1.0).astype(np.float32)
    al = al * (ok[:-1] & ok[1:])
    polyline_fade(img, pts, al, float(col[0]), float(col[1]), float(col[2]), width, glow)


def arrow3(img, p0, p1, cam, col, fov=45, width=2.0, glow=3.0, head=12.0):
    x, y, z = project(np.array([p0, p1], np.float64), cam, fov)
    if z.min() > 0.05:
        arrow(img, x[0], y[0], x[1], y[1], col, width, glow, head)


def bg_grid(img, plane, col=None, step=1.0, rng=8):
    col = C('#2a3d66', 0.10) if col is None else col
    for k in np.arange(-rng, rng + 1e-9, step):
        X, Y = plane(np.array([k, k]), np.array([-rng * 2, rng * 2]))
        line(img, X[0], Y[0], X[1], Y[1], float(col[0]), float(col[1]), float(col[2]), 1.0, 0.0)
        X, Y = plane(np.array([-rng, rng]), np.array([k, k]))
        line(img, X[0], Y[0], X[1], Y[1], float(col[0]), float(col[1]), float(col[2]), 1.0, 0.0)


def fill_poly(img, X, Y, col, alpha):
    import cv2
    m = np.zeros((H, W), np.uint8)
    cv2.fillPoly(m, [np.stack([X, Y], 1).astype(np.int32)], 1, lineType=cv2.LINE_AA)
    img += (m[..., None] * alpha).astype(np.float32) * col


def dashes_along(img, X, Y, col, phase, n_dash=10, duty=0.45, width=2.6, glow=5.0, heads=True):
    """Trazos que circulan a lo largo de una curva (orientación visible)."""
    pts = np.stack([X, Y], 1)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    Ltot = s[-1]
    if Ltot < 1e-6:
        return
    per = Ltot / n_dash
    for k in range(n_dash):
        a = ((k + phase) * per) % Ltot
        b = a + per * duty
        ss = np.linspace(a, b, 12) % Ltot
        xs = np.interp(ss, s, X); ys = np.interp(ss, s, Y)
        jump = np.abs(np.diff(ss)) > per
        if jump.any():
            continue
        pl(img, xs, ys, col, width, glow)
        if heads:
            arrow(img, xs[-3], ys[-3], xs[-1], ys[-1], col, width, glow, 12)


# ───────────────────────────── 1 · Teorema fundamental del cálculo ─────────────────────────────

class FTC(Scene):
    """∫ₐᵇ f'(x)dx = f(b) − f(a): el área bajo f' se acumula y dibuja f."""

    def draw(self, t, u):
        img = canvas()
        P = Plane(150, W * 0.5, H * 0.66)
        x = np.linspace(-3.0, 3.0, 400)
        fp = 0.9 * np.cos(1.3 * x) + 0.35 * np.sin(2.7 * x) + 0.25
        F = np.cumsum(fp) * (x[1] - x[0])
        F = F - F[np.searchsorted(x, -2.2)]
        rev = ease_out(min(u * 4, 1))
        ax = WHITE * 0.6 * rev
        X, Y = P([-3.2, 3.2], [0, 0]); line(img, X[0], Y[0], X[1], Y[1], *ax, 1.6, 2.0)
        X, Y = P([-3.2, -3.2], [-1.5, 4.0]); line(img, X[0], Y[0], X[1], Y[1], *ax, 1.6, 2.0)
        a, b = -2.2, 2.4
        xb = a + (b - a) * ease_in_out(min(max((u - 0.12) / 0.6, 0), 1))
        m = (x >= a) & (x <= xb)
        # área con signo: dorado positivo, azul negativo
        for xi, yi in zip(x[m][::2], fp[m][::2]):
            X, Y = P([xi, xi], [0, yi])
            c = GOLD if yi > 0 else BLUE
            line(img, X[0], Y[0], X[1], Y[1], *(c * 0.22), 2.4, 0.0)
        X, Y = P(x, fp); pl(img, X, Y, CYAN * 1.4 * rev, 2.6, 6.0)
        # f(x) acumulada hasta xb, desplazada hacia arriba
        off = 2.4
        mm = (x >= a) & (x <= xb)
        X, Y = P(x[mm], F[mm] + off); pl(img, X, Y, MAG * 1.5, 3.0, 8.0)
        if mm.sum() > 1:
            add_glow_point(img, X[-1], Y[-1], WHITE * 1.2, 5.0)
            Xc, Yc = P([xb, xb], [0, F[mm][-1] + off])
            line(img, Xc[0], Yc[0], Xc[1], Yc[1], *(WHITE * 0.25), 1.0, 0.0)
        for xv in (a, b):
            Xa, Ya = P([xv, xv], [-0.2, 0.2]); line(img, Xa[0], Ya[0], Xa[1], Ya[1], *(GOLD * 1.2), 2.4, 4.0)
        if u > 0.75:
            k = ease_out((u - 0.75) / 0.2)
            Fb = np.interp(b, x, F) + off; Fa = np.interp(a, x, F) + off
            Xb, Yb = P([b + 0.35, b + 0.35], [Fa, Fa + (Fb - Fa) * k])
            arrow(img, Xb[0], Yb[0], Xb[1], Yb[1], GOLD * 1.6, 3.0, 7.0, 18)
            Xh, Yh = P([a, b + 0.35], [Fa, Fa]); line(img, Xh[0], Yh[0], Xh[1], Yh[1], *(GOLD * 0.5), 1.2, 2.0)
        return img


# ───────────────────────────── 2 · Gradiente: curvas de nivel y caminos ─────────────────────────────

@njit(parallel=True, cache=True)
def _levels(out, cx, cy, s, cs, amps, sig, levels, t):
    h, w, _ = out.shape
    for j in prange(h):
        for i in range(w):
            x = (i - cx) / s; y = (cy - j) / s
            f = 0.0
            for k in range(cs.shape[0]):
                dx = x - cs[k, 0]; dy = y - cs[k, 1]
                f += amps[k] * math.exp(-(dx * dx + dy * dy) / (2 * sig[k] * sig[k]))
            v = f * levels + 0.5
            fr = v - math.floor(v)
            d = min(fr, 1 - fr)
            ln = math.exp(-d * d / 0.0018)
            hgt = min(max(f * 0.5 + 0.5, 0.0), 1.0)
            r = 0.15 + 0.85 * hgt; g = 0.35 + 0.35 * hgt; b = 1.0 - 0.75 * hgt
            ln *= min(abs(f) * 6.0, 1.0)
            out[j, i, 0] = ln * r * 0.55 + 0.012 * hgt
            out[j, i, 1] = ln * g * 0.55 + 0.006 * hgt
            out[j, i, 2] = ln * b * 0.55 + 0.02 * (1 - hgt)


class GradientPaths(Scene):
    """∫_γ ∇f·dr = f(q) − f(p): varios caminos entre p y q, mismo resultado."""
    scale = 0.5

    def prepare(self, dur):
        self.cs = np.array([[0.9, 1.6], [-1.0, -0.9], [1.2, -1.8], [-1.3, 2.4]])
        self.amps = np.array([1.2, -1.0, 0.7, -0.6])
        self.sig = np.array([0.9, 1.0, 0.7, 0.8])
        self.p = np.array([-1.0, -0.9]); self.q = np.array([0.9, 1.6])

    def f(self, x, y):
        v = 0
        for c, a, s in zip(self.cs, self.amps, self.sig):
            v = v + a * np.exp(-((x - c[0]) ** 2 + (y - c[1]) ** 2) / (2 * s * s))
        return v

    def draw(self, t, u):
        w, h = self.size()
        sm = np.zeros((h, w, 3), np.float32)
        S = 230 * self.scale
        _levels(sm, w / 2, h / 2, S, self.cs, self.amps, self.sig, 7.0, t)
        img = resize(sm, W, H) * ease_out(min(u * 3, 1))
        P = Plane(230)
        # campo gradiente en flechas finas
        gx, gy = np.meshgrid(np.linspace(-2.2, 2.2, 10), np.linspace(-3.8, 3.8, 17))
        e = 1e-3
        fx = (self.f(gx + e, gy) - self.f(gx - e, gy)) / (2 * e)
        fy = (self.f(gx, gy + e) - self.f(gx, gy - e)) / (2 * e)
        for x0, y0, a, b in zip(gx.ravel(), gy.ravel(), fx.ravel(), fy.ravel()):
            L = math.hypot(a, b) + 1e-9
            k = min(L, 1.2) * 0.22 / L
            X, Y = P([x0, x0 + a * k], [y0, y0 + b * k])
            arrow(img, X[0], Y[0], X[1], Y[1], WHITE * 0.28, 1.2, 0.0, 6)
        # tres caminos de p a q
        cols = [GOLD, CYAN, MAG]
        for k, (bend, col) in enumerate(zip((1.6, -1.2, 0.2), cols)):
            s = np.linspace(0, 1, 160)
            d = self.q - self.p
            nrm = np.array([-d[1], d[0]]) / np.linalg.norm(d)
            pts = self.p[None] + s[:, None] * d[None] + (np.sin(np.pi * s) * bend + 0.35 * np.sin(3 * np.pi * s) * (k - 1))[:, None] * nrm[None]
            tk = min(max((u - 0.1 - k * 0.12) / 0.4, 0), 1)
            n = max(int(160 * ease_in_out(tk)), 2)
            X, Y = P(pts[:n, 0], pts[:n, 1])
            pl(img, X, Y, col * 1.6, 3.0, 7.0)
            if tk < 1 and tk > 0:
                add_glow_point(img, X[-1], Y[-1], col * 1.5, 5.0)
        for pt, col in ((self.p, BLUE), (self.q, ORANGE)):
            X, Y = P([pt[0]], [pt[1]])
            add_glow_point(img, X[0], Y[0], col * 2.0, 7.0)
        return img


# ───────────────────────────── 3 · Campos vectoriales con partículas ─────────────────────────────

def field(kind, x, y, t=0.0):
    if kind == 'curl':
        r2 = x * x + y * y + 0.15
        return -y / r2 * 1.6 + 0.0 * x, x / r2 * 1.6
    if kind == 'source':
        r2 = x * x + y * y + 0.05
        return x / r2, y / r2
    if kind == 'saddle':
        return x, -y
    if kind == 'dipole':
        ex = 0.0; ey = 0.0
        for (qx, qy, q) in ((0.0, 1.1, 1.0), (0.0, -1.1, -1.0)):
            dx = x - qx; dy = y - qy
            r3 = (dx * dx + dy * dy + 0.02) ** 1.5
            ex = ex + q * dx / r3; ey = ey + q * dy / r3
        return ex, ey
    if kind == 'shear':
        return np.sin(y * 1.2) * 1.2, 0.0 * x
    if kind == 'swirl':
        return -y + 0.3 * x * np.cos(t), x + 0.3 * y * np.sin(t)
    raise KeyError(kind)


class FieldFlow(Scene):
    """Partículas que siguen un campo vectorial, con estelas; opcionalmente una curva o una caja de flujo."""

    def prepare(self, dur):
        rng = self.rng
        kind = self.get('kind', 'curl')
        n = self.get('n', 1800)
        R = self.get('extent', (2.5, 4.4))
        self.R = R
        pos = np.stack([rng.uniform(-R[0], R[0], n), rng.uniform(-R[1], R[1], n)], 1)
        frames = int(dur * FPS) + 2
        trail = self.get('trail', 10)
        self.hist = np.zeros((frames + trail, n, 2), np.float32)
        dt = self.get('speed', 0.9) / FPS
        life = rng.integers(20, 90, n)
        for f in range(frames + trail):
            vx, vy = field(kind, pos[:, 0], pos[:, 1], f / FPS)
            vv = np.sqrt(vx ** 2 + vy ** 2) + 1e-9
            k = np.minimum(vv, self.get('vmax', 2.5)) / vv
            pos[:, 0] += vx * k * dt; pos[:, 1] += vy * k * dt
            life -= 1
            out = (np.abs(pos[:, 0]) > R[0] * 1.1) | (np.abs(pos[:, 1]) > R[1] * 1.1) | (life <= 0)
            if kind == 'dipole':
                out |= np.hypot(pos[:, 0], pos[:, 1] + 1.1) < 0.12
            m = out.sum()
            if m:
                pos[out] = np.stack([rng.uniform(-R[0], R[0], m), rng.uniform(-R[1], R[1], m)], 1)
                if kind == 'source':
                    a = rng.random(m) * 2 * np.pi; r = rng.uniform(0.05, 0.4, m)
                    pos[out] = np.stack([r * np.cos(a), r * np.sin(a)], 1)
                if kind == 'dipole':
                    a = rng.random(m) * 2 * np.pi
                    pos[out] = np.stack([0.15 * np.cos(a), 1.1 + 0.15 * np.sin(a)], 1)
                life[out] = rng.integers(20, 90, m)
            self.hist[f] = pos
        self.trail = trail
        self.kind = kind
        self.reset = np.zeros(n, bool)

    def draw(self, t, u):
        img = canvas()
        P = Plane(self.get('zoom', 210))
        bg_grid(img, P, step=0.5, rng=6)
        f = min(int(t * FPS), self.hist.shape[0] - self.trail - 1) + self.trail
        col = C(self.get('color', '#53e0ff'))
        seg = self.hist[f - self.trail:f + 1]
        X, Y = P(seg[..., 0], seg[..., 1])
        n = seg.shape[1]
        jumps = np.abs(np.diff(seg, axis=0)).max(axis=2) > 0.5
        for i in range(n):
            if jumps[:, i].any():
                continue
            al = np.linspace(0.0, 1.0, self.trail, dtype=np.float32) ** 1.6 * 0.9
            pts = np.stack([X[:, i], Y[:, i]], 1).astype(np.float32)
            polyline_fade(img, pts, al, float(col[0]), float(col[1]), float(col[2]), 1.5, 2.0)
        overlay = self.get('overlay', None)
        if overlay == 'loop':
            th = np.linspace(0, 2 * np.pi, 300)
            r = self.get('loop_r', 1.4)
            cx, cy = self.get('loop_c', (0.0, 0.0))
            X, Y = P(cx + r * np.cos(th), cy + r * np.sin(th))
            pl(img, X, Y, GOLD * 0.7, 2.0, 5.0)
            dashes_along(img, X, Y, GOLD * 1.6, t * 0.8, 10)
        elif overlay == 'box':
            s = self.get('box_s', 1.0)
            k = 0.5 + 0.5 * math.sin(t * 2)
            sq = np.array([[-s, -s], [s, -s], [s, s], [-s, s], [-s, -s]])
            X, Y = P(sq[:, 0], sq[:, 1])
            pl(img, X, Y, GOLD * 1.3, 2.4, 6.0)
            for (x0, y0, nx, ny) in ((s, 0, 1, 0), (-s, 0, -1, 0), (0, s, 0, 1), (0, -s, 0, -1)):
                Xa, Ya = P([x0, x0 + nx * 0.45], [y0, y0 + ny * 0.45])
                arrow(img, Xa[0], Ya[0], Xa[1], Ya[1], ORANGE * 1.6, 2.6, 6.0, 14)
        elif overlay == 'charges':
            for (qy, c) in ((1.1, ORANGE), (-1.1, BLUE)):
                X, Y = P([0.0], [qy]); add_glow_point(img, X[0], Y[0], c * 3.0, 9.0)
        elif overlay == 'paddle':
            # rueda de paletas que gira: interpretación del rotacional
            cx, cy = 0.9, 0.0
            ang = t * 2.4
            for k in range(4):
                a = ang + k * np.pi / 2
                X, Y = P([cx, cx + 0.45 * np.cos(a)], [cy, cy + 0.45 * np.sin(a)])
                line(img, X[0], Y[0], X[1], Y[1], *(MAG * 1.8), 3.0, 6.0)
            X, Y = P([cx], [cy]); add_glow_point(img, X[0], Y[0], MAG * 2.0, 5.0)
        return img


# ───────────────────────────── 4 · Green: cancelación interior ─────────────────────────────

class GreenTiling(Scene):
    """La región se trocea en celdas con circulación propia; los lados interiores se cancelan."""

    def region_r(self, th):
        return 1.75 + 0.35 * np.sin(3 * th + 0.4) + 0.18 * np.cos(5 * th)

    def inside(self, x, y):
        r = np.hypot(x, y); th = np.arctan2(y, x)
        return r < self.region_r(th)

    def draw(self, t, u):
        img = canvas()
        P = Plane(205)
        bg_grid(img, P, step=0.5, rng=6)
        levels = self.get('levels', (3, 6, 12, 24))
        kidx = min(int(u * len(levels) * 1.0), len(levels) - 1)
        n = levels[kidx]
        local = (u * len(levels)) % 1.0
        L = 2.3
        h = 2 * L / n
        xs = -L + (np.arange(n) + 0.5) * h
        cells = [(xs[i], xs[j], i, j) for i in range(n) for j in range(n) if self.inside(xs[i], xs[j])]
        cellset = set((i, j) for _, _, i, j in cells)
        phase = t * 1.5
        # circulación de cada celda: pequeños bucles antihorarios
        for (x, y, _, _) in cells:
            s = h * 0.5 * 0.82
            sq = np.array([[x - s, y - s], [x + s, y - s], [x + s, y + s], [x - s, y + s], [x - s, y - s]])
            X, Y = P(sq[:, 0], sq[:, 1])
            pl(img, X, Y, CYAN * (0.25 if n > 8 else 0.4), 1.2, 1.5)
            if n <= 12:
                dashes_along(img, X, Y, CYAN * 0.9, phase, 4, 0.35, 1.6, 2.0, heads=n <= 6)
        # frontera de la unión de celdas (lo único que sobrevive)
        for (x, y, i, j) in cells:
            s = h * 0.5
            for (di, dj, a, b) in ((1, 0, (x + s, y - s), (x + s, y + s)), (-1, 0, (x - s, y + s), (x - s, y - s)),
                                   (0, 1, (x + s, y + s), (x - s, y + s)), (0, -1, (x - s, y - s), (x + s, y - s))):
                if (i + di, j + dj) not in cellset:
                    X, Y = P([a[0], b[0]], [a[1], b[1]])
                    line(img, X[0], Y[0], X[1], Y[1], *(GOLD * 1.5), 2.6, 6.0)
        # curva suave ∂D que emerge al final
        th = np.linspace(0, 2 * np.pi, 400)
        r = self.region_r(th)
        X, Y = P(r * np.cos(th), r * np.sin(th))
        k = ease_in_out(min(max((u - 0.55) / 0.4, 0), 1))
        pl(img, X, Y, WHITE * (0.15 + 1.2 * k), 2.0 + 1.0 * k, 3.0 + 6 * k)
        if k > 0.2:
            dashes_along(img, X, Y, GOLD * 1.8 * k, t * 0.6, 9, 0.4, 3.0, 7.0)
        return img


# ───────────────────────────── 5 · Superficies 3D, parametrizaciones y variedades ─────────────────────────────

def surf(kind, uu, vv, h=1.0):
    """Parametrizaciones φ(u, v) con (u, v) ∈ [0,1]²."""
    if kind == 'dome':
        r = uu * 1.0; th = vv * 2 * np.pi
        return np.stack([r * np.cos(th) * 1.6, h * (1 - r ** 2) * 1.4, r * np.sin(th) * 1.6], -1)
    if kind == 'sphere':
        th = vv * 2 * np.pi; ph = uu * np.pi
        return np.stack([np.sin(ph) * np.cos(th), np.cos(ph), np.sin(ph) * np.sin(th)], -1) * 1.5
    if kind == 'torus':
        a = uu * 2 * np.pi; b = vv * 2 * np.pi
        R, r = 1.3, 0.5
        return np.stack([(R + r * np.cos(b)) * np.cos(a), r * np.sin(b), (R + r * np.cos(b)) * np.sin(a)], -1)
    if kind == 'saddle':
        x = (uu - 0.5) * 3; z = (vv - 0.5) * 3
        return np.stack([x, 0.35 * (x * x - z * z), z], -1)
    if kind == 'mobius':
        a = uu * 2 * np.pi; s = (vv - 0.5) * 0.9
        return np.stack([(1.3 + s * np.cos(a / 2)) * np.cos(a), s * np.sin(a / 2), (1.3 + s * np.cos(a / 2)) * np.sin(a)], -1)
    if kind == 'wave':
        x = (uu - 0.5) * 3.2; z = (vv - 0.5) * 3.2
        return np.stack([x, 0.35 * np.sin(2 * x) * np.cos(1.6 * z), z], -1)
    raise KeyError(kind)


class Parametrization(Scene):
    """Un dominio plano D ⊂ ℝ² se deforma y se convierte en la variedad M = φ(D)."""

    def draw(self, t, u):
        img = canvas()
        kind = self.get('kind', 'torus')
        ang = self.get('yaw', 0.6) + self.get('orbit', 0.35) * t
        el = self.get('elev', 0.55)
        d = self.get('dist', 6.5)
        cam = look_at((math.sin(ang) * d * math.cos(el), d * math.sin(el), -math.cos(ang) * d * math.cos(el)), (0, 0, 0))
        fov = 45
        m0, m1 = self.get('morph', (0.08, 0.6))
        s = ease_in_out(min(max((u - m0) / (m1 - m0), 0), 1))
        n = self.get('grid', 24)
        uu = np.linspace(0, 1, 120)

        def pts(U, V):
            flat = np.stack([(U - 0.5) * 3.4, np.full_like(U, -1.6), (V - 0.5) * 3.4], -1)
            return flat * (1 - s) + surf(kind, U, V) * s

        for k in np.linspace(0, 1, n + 1):
            for orient in (0, 1):
                U = np.full_like(uu, k) if orient == 0 else uu
                V = uu if orient == 0 else np.full_like(uu, k)
                col = CYAN if orient == 0 else VIO
                pl3(img, pts(U, V), cam, col * 0.9, fov, 1.2, 2.0, depth_fade=(d - 3, d + 3))
        # celda resaltada con los vectores tangentes φ_u, φ_v y la normal
        cu, cv = self.get('cell', (0.35, 0.3))
        du = 1.0 / n
        p0 = pts(np.array(cu), np.array(cv))
        pu = pts(np.array(cu + du), np.array(cv)); pv = pts(np.array(cu), np.array(cv + du))
        quad = np.array([p0, pu, pts(np.array(cu + du), np.array(cv + du)), pv, p0])
        pl3(img, quad, cam, GOLD * 2.0, fov, 2.6, 6.0)
        tu = (pu - p0) * 2.5; tv = (pv - p0) * 2.5
        arrow3(img, p0, p0 + tu, cam, ORANGE * 1.8, fov, 2.6, 5.0)
        arrow3(img, p0, p0 + tv, cam, GREEN * 1.6, fov, 2.6, 5.0)
        nn = np.cross(tu, tv)
        if np.linalg.norm(nn) > 1e-6 and s > 0.5:
            nn = nn / np.linalg.norm(nn) * 0.8
            arrow3(img, p0, p0 + nn, cam, MAG * 1.8, fov, 2.6, 6.0)
        return img


class Charts(Scene):
    """Atlas: parches planos que vuelan desde ℝ² y se pegan sobre la esfera / toro (cartas solapadas)."""

    def draw(self, t, u):
        img = canvas()
        kind = self.get('kind', 'sphere')
        ang = 0.4 + 0.3 * t
        d = 6.8
        cam = look_at((math.sin(ang) * d * 0.85, d * 0.45, -math.cos(ang) * d * 0.85), (0, 0, 0))
        fov = 45
        uu = np.linspace(0, 1, 100)
        # malla fantasma de la variedad
        for k in np.linspace(0, 1, 19):
            pl3(img, surf(kind, np.full_like(uu, k), uu), cam, WHITE * 0.12, fov, 1.0, 0.0)
            pl3(img, surf(kind, uu, np.full_like(uu, k)), cam, WHITE * 0.12, fov, 1.0, 0.0)
        patches = [((0.15, 0.45), (0.05, 0.4), CYAN), ((0.35, 0.7), (0.3, 0.65), GOLD), ((0.55, 0.9), (0.55, 0.95), MAG),
                   ((0.2, 0.6), (0.6, 0.98), GREEN)]
        for k, ((u0, u1), (v0, v1), col) in enumerate(patches):
            tk = ease_in_out(min(max((u - 0.08 - k * 0.14) / 0.3, 0), 1))
            offs = np.array([3.5 * math.cos(k * 1.7), 2.5, 3.5 * math.sin(k * 1.7)])
            g = np.linspace(0, 1, 7)
            for a in g:
                for orient in (0, 1):
                    U = u0 + (u1 - u0) * (np.full_like(uu, a) if orient == 0 else uu)
                    V = v0 + (v1 - v0) * (uu if orient == 0 else np.full_like(uu, a))
                    on = surf(kind, U, V) * 1.01
                    flat = np.stack([(U - (u0 + u1) / 2) * 3, np.zeros_like(U), (V - (v0 + v1) / 2) * 3], -1) + offs
                    P = flat * (1 - tk) + on * tk
                    pl3(img, P, cam, col * 1.3, fov, 1.4, 3.0)
        return img


class StokesNet(Scene):
    """Superficies distintas con el mismo borde ∂S: el flujo del rotacional no cambia."""

    def draw(self, t, u):
        img = canvas()
        ang = 0.5 + 0.35 * t
        d = 6.4
        cam = look_at((math.sin(ang) * d * 0.8, d * 0.55, -math.cos(ang) * d * 0.8), (0, 0.3, 0))
        fov = 45
        h = 1.25 * math.sin(t * self.get('freq', 1.6)) + 0.15
        uu = np.linspace(0, 1, 100)
        n = 16
        for k in np.linspace(0, 1, n + 1):
            pl3(img, surf('dome', np.full_like(uu, k), uu, h), cam, CYAN * 0.8, fov, 1.2, 2.0, depth_fade=(d - 3, d + 3))
            pl3(img, surf('dome', uu, np.full_like(uu, k), h), cam, CYAN * 0.5, fov, 1.0, 1.5, depth_fade=(d - 3, d + 3))
        # pequeños bucles de circulación sobre la superficie
        for a in np.linspace(0.2, 0.8, 4):
            for b in np.linspace(0, 1, 9, endpoint=False):
                c = surf('dome', np.array(a), np.array(b + 0.03), h)
                th = np.linspace(0, 2 * np.pi, 20) + t * 3
                ring = np.array([surf('dome', np.array(a + 0.045 * math.cos(q)), np.array(b + 0.03 + 0.02 * math.sin(q)), h) for q in th])
                pl3(img, ring, cam, MAG * 0.9, fov, 1.2, 2.0)
        # normales
        for b in np.linspace(0, 1, 8, endpoint=False):
            p = surf('dome', np.array(0.5), np.array(b), h)
            e = 1e-3
            pu = surf('dome', np.array(0.5 + e), np.array(b), h) - p
            pv = surf('dome', np.array(0.5), np.array(b + e), h) - p
            nn = np.cross(pv, pu); nn /= np.linalg.norm(nn) + 1e-9
            arrow3(img, p, p + nn * 0.55, cam, ORANGE * 1.4, fov, 1.8, 3.0, 10)
        # borde: circunferencia fija con flechas de orientación
        th = np.linspace(0, 2 * np.pi, 300)
        B = np.stack([1.6 * np.cos(th), np.zeros_like(th), 1.6 * np.sin(th)], 1)
        pl3(img, B, cam, GOLD * 1.8, fov, 3.2, 8.0)
        x, y, z = project(B, cam, fov)
        dashes_along(img, x, y, GOLD * 2.2, t * 0.7, 8, 0.3, 3.4, 8.0)
        return img


class GaussSphere(Scene):
    """Flujo saliente a través de una esfera: lo que nace dentro (divergencia) atraviesa el borde."""

    def prepare(self, dur):
        rng = self.rng
        n = 900
        v = rng.normal(size=(n, 3)); v /= np.linalg.norm(v, axis=1, keepdims=True)
        self.dirs = v; self.ph = rng.random(n); self.spd = rng.uniform(0.5, 1.0, n)
        fib = np.arange(140) + 0.5
        phi = np.arccos(1 - 2 * fib / 140); th = np.pi * (1 + 5 ** 0.5) * fib
        self.normals = np.stack([np.cos(th) * np.sin(phi), np.cos(phi), np.sin(th) * np.sin(phi)], 1)

    def draw(self, t, u):
        img = canvas()
        ang = 0.3 + 0.3 * t
        d = 7.0
        cam = look_at((math.sin(ang) * d * 0.85, d * 0.4, -math.cos(ang) * d * 0.85), (0, 0, 0))
        fov = 45
        R = 1.6
        uu = np.linspace(0, 1, 120)
        for k in np.linspace(0.08, 0.92, 11):
            pl3(img, surf('sphere', np.full_like(uu, k), uu) * (R / 1.5), cam, CYAN * 0.55, fov, 1.1, 2.0, depth_fade=(d - 2, d + 2))
        for k in np.linspace(0, 1, 16, endpoint=False):
            pl3(img, surf('sphere', uu, np.full_like(uu, k)) * (R / 1.5), cam, CYAN * 0.35, fov, 1.0, 1.5, depth_fade=(d - 2, d + 2))
        # partículas radiales desde el interior
        r = ((self.ph + t * 0.35 * self.spd) % 1.0) * 3.2
        P = self.dirs * r[:, None]
        x, y, z = project(P, cam, fov)
        inside = r < R
        cols = np.where(inside[:, None], GOLD * 0.9, ORANGE * 0.6) * np.clip(1.4 - r / 3.2, 0, 1)[:, None]
        splat(img, x.astype(np.float32), y.astype(np.float32), (cols * 10).astype(np.float32), np.full(len(x), 1.2, np.float32))
        # vectores normales exteriores
        grow = ease_out(min(u * 2, 1))
        for nv in self.normals[::2]:
            p = nv * R
            arrow3(img, p, p + nv * 0.6 * grow, cam, ORANGE * 1.1, fov, 1.5, 2.5, 8)
        x, y, _ = project(np.zeros((1, 3)), cam, fov)
        add_glow_point(img, x[0], y[0], GOLD * 2.5, 10.0)
        return img


# ───────────────────────────── 6 · Álgebra exterior ─────────────────────────────

class OneForm(Scene):
    """Una 1-forma como pila de líneas de nivel: ω(v) = número de hojas que atraviesa v."""

    def draw(self, t, u):
        img = canvas()
        P = Plane(200)
        kind = self.get('kind', 'constant')
        yy, xx = np.mgrid[0:H:2, 0:W:2]
        x = (xx - W / 2) / 200.0; y = (H / 2 - yy) / 200.0
        if kind == 'constant':
            a = 0.5 + 0.2 * math.sin(t * 0.7)
            f = (np.cos(a) * x + np.sin(a) * y) * 2.4
        else:
            f = 0.9 * (x * x - 0.6 * y * y) + 0.6 * np.sin(1.5 * y + t * 0.5)
        fr = f - np.floor(f)
        d = np.minimum(fr, 1 - fr)
        lines = np.exp(-d * d / 0.0009).astype(np.float32)
        band = (np.floor(f) % 2 == 0).astype(np.float32) * 0.03
        lay = np.stack([lines * 0.35 + band * 0.5, lines * 0.45 + band * 0.4, lines * 0.9 + band], -1)
        img += resize(lay, W, H)
        # vector v que gira y cruza hojas
        ang = t * 1.2
        L = 1.6 + 0.6 * math.sin(t * 0.9)
        X, Y = P([0, L * math.cos(ang)], [0, L * math.sin(ang)])
        arrow(img, X[0], Y[0], X[1], Y[1], GOLD * 2.2, 3.4, 8.0, 22)
        add_glow_point(img, X[0], Y[0], WHITE * 1.5, 6.0)
        return img


class Wedge(Scene):
    """dxⁱ∧dxʲ(u, v): áreas orientadas del paralelogramo proyectadas en los planos coordenados."""

    def draw(self, t, u):
        img = canvas()
        ang = 0.6 + 0.3 * t
        d = 7.5
        cam = look_at((math.sin(ang) * d * 0.8, d * 0.5, -math.cos(ang) * d * 0.8), (0.6, 0.5, 0.6))
        fov = 45
        uvec = np.array([1.8, 0.4 + 0.6 * math.sin(t), 0.3])
        vvec = np.array([0.3 + 0.8 * math.cos(t * 0.8), 1.6, 0.9])
        if self.get('swap', False) and u > 0.5:
            uvec, vvec = vvec, uvec
        O = np.zeros(3)
        for e, c in ((np.array([3, 0, 0]), WHITE), (np.array([0, 3, 0]), WHITE), (np.array([0, 0, 3]), WHITE)):
            arrow3(img, O, e, cam, c * 0.5, fov, 1.4, 1.5, 10)
        par = np.array([O, uvec, uvec + vvec, vvec, O])
        x, y, _ = project(par, cam, fov)
        sgn = np.cross(uvec, vvec)
        fill_poly(img, x, y, GOLD, 0.16)
        pl3(img, par, cam, GOLD * 1.8, fov, 2.8, 7.0)
        arrow3(img, O, uvec, cam, ORANGE * 2.0, fov, 3.0, 6.0, 16)
        arrow3(img, O, vvec, cam, GREEN * 1.8, fov, 3.0, 6.0, 16)
        # proyecciones sobre los tres planos (color según el signo del determinante)
        for k, (drop, comp) in enumerate(((1, 2), (2, 0), (0, 1))):
            Q = par.copy(); Q[:, drop] = 0.0 if k != 1 else Q[:, drop] * 0
            Q[:, drop] = 0
            area = sgn[[0, 1, 2][drop]]
            col = CYAN if area > 0 else MAG
            x, y, _ = project(Q, cam, fov)
            fill_poly(img, x, y, col, 0.10)
            pl3(img, Q, cam, col * 1.2, fov, 1.6, 3.0)
            for p, q in zip(par[:4], Q[:4]):
                pl3(img, np.array([p, q]), cam, WHITE * 0.12, fov, 1.0, 0.0)
        return img


class BoundaryCube(Scene):
    """∂∂M = ∅: las caras del cubo inducen orientaciones opuestas en cada arista y se cancelan."""

    def draw(self, t, u):
        img = canvas()
        ang = 0.7 + 0.4 * t
        d = 6.0
        cam = look_at((math.sin(ang) * d * 0.8, d * 0.55, -math.cos(ang) * d * 0.8), (0, 0, 0))
        fov = 45
        s = 1.0
        V = np.array([[x, y, z] for x in (-s, s) for y in (-s, s) for z in (-s, s)], float)
        faces = []
        for ax in range(3):
            for sg in (-1, 1):
                c = np.zeros(3); c[ax] = sg * s
                a1, a2 = [i for i in range(3) if i != ax]
                e1 = np.zeros(3); e1[a1] = s; e2 = np.zeros(3); e2[a2] = s
                if sg * (1 if (ax == 1) else 1) < 0:
                    e1, e2 = e2, e1
                if ax == 1:
                    e1, e2 = e2, e1
                faces.append((c, e1, e2))
        cancel = ease_in_out(min(max((u - 0.45) / 0.35, 0), 1))
        explode = 0.55 * (1 - ease_in_out(min(max((u - 0.15) / 0.3, 0), 1)))
        cols = [CYAN, MAG, GOLD, GREEN, VIO, ORANGE]
        for (c, e1, e2), col in zip(faces, cols):
            cen = c * (1 + explode)
            ins = 0.86
            corners = [cen + (-e1 - e2) * ins, cen + (e1 - e2) * ins, cen + (e1 + e2) * ins, cen + (-e1 + e2) * ins]
            loop = np.array(corners + [corners[0]])
            x, y, z = project(loop, cam, fov)
            fill_poly(img, x, y, col, 0.05 * (1 - cancel))
            pl3(img, loop, cam, col * (1.4 * (1 - cancel) + 0.08), fov, 2.0, 4.0)
            if cancel < 0.95:
                dashes_along(img, x, y, col * 1.8 * (1 - cancel), t * 1.2, 4, 0.3, 2.4, 5.0)
        # aristas del cubo como "fantasma" que queda
        edges = [(i, j) for i in range(8) for j in range(i + 1, 8) if np.sum(np.abs(V[i] - V[j])) == 2 * s]
        for i, j in edges:
            pl3(img, np.array([V[i], V[j]]), cam, WHITE * (0.08 + 0.4 * cancel), fov, 1.4, 3.0 * cancel)
        return img


class Pullback(Scene):
    """φ*: la rejilla (u, v) se deforma en (x, y); cada celda escala por det Dφ (el jacobiano aparece solo)."""

    def draw(self, t, u):
        img = canvas()
        Pp = Plane(230)
        s = 0.5 - 0.5 * math.cos(min(u * 1.3, 1.0) * math.pi)
        n = 16
        g = np.linspace(-1, 1, 200)

        def phi(U, V):
            # mapa polar suavizado mezclado con identidad
            r = 0.9 + 0.75 * (V + 1)
            th = U * 1.4 + 0.3 * np.sin(2 * V + t)
            X = r * np.sin(th); Y = r * np.cos(th) - 1.6
            return U * 1.6 * (1 - s) + X * s, V * 1.6 * (1 - s) + Y * s

        # mapa de calor del jacobiano
        uu, vv = np.meshgrid(np.linspace(-1, 1, 60), np.linspace(-1, 1, 60))
        e = 1e-3
        xu = (phi(uu + e, vv)[0] - phi(uu - e, vv)[0]) / (2 * e); yu = (phi(uu + e, vv)[1] - phi(uu - e, vv)[1]) / (2 * e)
        xv = (phi(uu, vv + e)[0] - phi(uu, vv - e)[0]) / (2 * e); yv = (phi(uu, vv + e)[1] - phi(uu, vv - e)[1]) / (2 * e)
        J = xu * yv - xv * yu
        for k in np.linspace(-1, 1, n + 1):
            X, Y = phi(np.full_like(g, k), g); X, Y = Pp(X, Y); pl(img, X, Y, CYAN * 0.9, 1.4, 2.0)
            X, Y = phi(g, np.full_like(g, k)); X, Y = Pp(X, Y); pl(img, X, Y, VIO * 0.9, 1.4, 2.0)
        # celdas coloreadas según J
        for i in range(0, n, 2):
            for j in range(0, n, 3):
                u0 = -1 + 2 * i / n; v0 = -1 + 2 * j / n; du = 2 / n
                Uq = np.array([u0, u0 + du, u0 + du, u0]); Vq = np.array([v0, v0, v0 + du, v0 + du])
                X, Y = phi(Uq, Vq); X, Y = Pp(X, Y)
                jj = np.interp(i, [0, n], [0, 59])
                Jv = J[int(np.clip((v0 + 1) / 2 * 59, 0, 59)), int(np.clip((u0 + 1) / 2 * 59, 0, 59))]
                col = GOLD * min(abs(Jv) / 3.0, 1.5) + C('#000000')
                fill_poly(img, X, Y, col, 0.25)
        # celda destacada con los vectores columna de Dφ
        u0, v0 = 0.2, -0.1
        X0, Y0 = phi(np.array(u0), np.array(v0))
        Xu, Yu = phi(np.array(u0 + 0.25), np.array(v0)); Xv, Yv = phi(np.array(u0), np.array(v0 + 0.25))
        a0 = Pp(X0, Y0); a1 = Pp(Xu, Yu); a2 = Pp(Xv, Yv)
        arrow(img, a0[0], a0[1], a1[0], a1[1], ORANGE * 2.0, 3.0, 6.0, 16)
        arrow(img, a0[0], a0[1], a2[0], a2[1], GREEN * 2.0, 3.0, 6.0, 16)
        return img


class EllipseArea(Scene):
    """Área = ½∮(x dy − y dx): el radio vector barre la elipse (πab)."""

    def draw(self, t, u):
        img = canvas()
        P = Plane(220)
        bg_grid(img, P, step=0.5, rng=6)
        a, b = 2.1, 1.3
        k = ease_in_out(min(max((u - 0.05) / 0.75, 0), 1))
        th = np.linspace(0, 2 * np.pi * k, max(int(400 * k), 2))
        X, Y = P(a * np.cos(th), b * np.sin(th))
        if len(th) > 2:
            Xo, Yo = P(0, 0)
            fill_poly(img, np.concatenate([[Xo], X]), np.concatenate([[Yo], Y]), GOLD, 0.16)
        thf = np.linspace(0, 2 * np.pi, 400)
        Xf, Yf = P(a * np.cos(thf), b * np.sin(thf))
        pl(img, Xf, Yf, WHITE * 0.2, 1.2, 0.0)
        pl(img, X, Y, GOLD * 1.8, 3.0, 7.0)
        if len(th) > 1:
            Xo, Yo = P(0, 0)
            arrow(img, Xo, Yo, X[-1], Y[-1], CYAN * 1.8, 2.6, 6.0, 16)
            # velocidad (tangente)
            tt = th[-1]
            Xt, Yt = P([a * math.cos(tt), a * math.cos(tt) - 0.6 * a * math.sin(tt)], [b * math.sin(tt), b * math.sin(tt) + 0.6 * b * math.cos(tt)])
            arrow(img, Xt[0], Yt[0], Xt[1], Yt[1], MAG * 1.8, 2.4, 5.0, 14)
        return img


class FormulaVoid(Scene):
    """Fondo casi vacío para fórmulas a pantalla completa: rejilla tenue y partículas lentas."""

    def prepare(self, dur):
        n = 500
        self.x = self.rng.random(n); self.y = self.rng.random(n); self.z = self.rng.uniform(0.3, 1.0, n)

    def draw(self, t, u):
        img = canvas()
        P = Plane(120 + 30 * u)
        bg_grid(img, P, C('#2a3d66', 0.08), step=1.0, rng=10)
        x = ((self.x + t * 0.01 * self.z) % 1) * W; y = ((self.y - t * 0.02 * self.z) % 1) * H
        splat(img, x.astype(np.float32), y.astype(np.float32), (np.ones((len(x), 3)) * self.z[:, None] * 0.6).astype(np.float32),
              np.full(len(x), 1.0, np.float32))
        return img
