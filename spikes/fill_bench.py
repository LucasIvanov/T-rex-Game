"""Spike (descartavel): mede o custo de preencher a cena por scanline em Python + numpy.

Cena parecida com a do jogo: terreno, 1 cubo e 4 esferas (8x6 = 48 faces) em projecao
ortografica com elevacao de 20 graus. Sol = luz pontual. Modos: constante, Gouraud, Phong.
Estrategias de preenchimento (mesma varredura por interseccao de arestas, so muda o laco):
  A = um trecho (span) por vez em Python
  B = todas as linhas de um poligono de uma vez
  C = todos os poligonos da cena de uma vez (ordem do pintor preservada)

Uso: python fill_bench.py [pasta_para_png]
"""
import os
import sys
import time
import math

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import numpy as np
import pygame

W, H = 800, 450
ELEV = math.radians(20)
COSE, SINE = math.cos(ELEV), math.sin(ELEV)
S = 55.0
CX, CY = 400.0, 300.0
SUN = np.array([3.0, 9.0, 2.0], dtype=np.float32)
VIEW = np.array([0.0, SINE, COSE], dtype=np.float32)
KA, KD, KS, SHINE = 0.25, 0.75, 0.5, 30.0
MATS = np.array([[0.80, 0.55, 0.30], [0.75, 0.75, 0.78], [0.30, 0.55, 0.20]], dtype=np.float32)
SKY = np.array([120, 180, 235], dtype=np.uint8)


# ---------------------------------------------------------------- malhas
class Mesh:
    def __init__(self, verts, vnorm, faces, mat):
        self.v = np.asarray(verts, np.float32)
        self.n = np.asarray(vnorm, np.float32)
        self.f = np.asarray(faces, np.int32)
        self.mat = mat
        fn = self.n[self.f].mean(axis=1)
        self.fn = (fn / np.linalg.norm(fn, axis=1, keepdims=True)).astype(np.float32)


def make_cube(side, mat):
    h = side / 2
    verts, norms, faces = [], [], []
    for axis in range(3):
        for sign in (1, -1):
            u, v = [a for a in range(3) if a != axis]
            nrm = np.zeros(3)
            nrm[axis] = sign
            for du, dv in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                p = np.zeros(3)
                p[axis] = sign * h
                p[u] = du * h
                p[v] = dv * h
                verts.append(p)
                norms.append(nrm)
            b = len(verts) - 4
            faces.append([b, b + 1, b + 2, b + 3])
    return Mesh(verts, norms, faces, mat)


def make_sphere(r, stacks, slices, mat):
    verts = [[0, r, 0]]
    for s in range(1, stacks):
        phi = math.pi * s / stacks
        for j in range(slices):
            th = 2 * math.pi * j / slices
            verts.append([r * math.sin(phi) * math.cos(th), r * math.cos(phi), r * math.sin(phi) * math.sin(th)])
    verts.append([0, -r, 0])
    verts = np.array(verts)
    norms = verts / r
    last = len(verts) - 1
    ring = lambda s, j: 1 + (s - 1) * slices + (j % slices)
    faces = []
    for j in range(slices):
        faces.append([0, ring(1, j), ring(1, j + 1), ring(1, j + 1)])
    for s in range(1, stacks - 1):
        for j in range(slices):
            faces.append([ring(s, j), ring(s + 1, j), ring(s + 1, j + 1), ring(s, j + 1)])
    for j in range(slices):
        faces.append([last, ring(stacks - 1, j + 1), ring(stacks - 1, j), ring(stacks - 1, j)])
    return Mesh(verts, norms, faces, mat)


def make_ground(mat):
    v = [[-8, 0, -6], [8, 0, -6], [8, 0, 8], [-8, 0, 8]]
    return Mesh(v, [[0, 1, 0]] * 4, [[0, 1, 2, 3]], mat)


GROUND = make_ground(0)
CUBE = make_cube(1.2, 1)
SPHERE = make_sphere(0.6, 6, 8, 2)


# ---------------------------------------------------------------- cena
def instance_faces(mesh, pos, theta):
    c, s = math.cos(theta), math.sin(theta)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], np.float32)
    wv = mesh.v @ R.T + np.asarray(pos, np.float32)
    wn = mesh.n @ R.T
    fn = mesh.fn @ R.T
    vis = (fn @ VIEW) > 1e-6
    idx = mesh.f[vis]
    return wv[idx], wn[idx], fn[vis], np.full(idx.shape[0], mesh.mat, np.int32)


def project(PV):
    x, y, z = PV[..., 0], PV[..., 1], PV[..., 2]
    return np.stack([CX + S * x, CY - S * (y * COSE - z * SINE)], -1).astype(np.float64)


def build_scene(t):
    """Devolve (grupo_terreno, grupo_objetos); cada grupo = (P_tela, PV, NV, FN, mat)."""
    x0 = np.array([-1.0, 2.2, 5.0, 7.2])
    xs = ((x0 - 3.0 * t + 8) % 16) - 8
    items = []
    jump = 1.5 * abs(math.sin(math.pi * t / 1.2))
    items.append((CUBE, (-5.5, 0.6 + jump, 0.0), 0.0))
    for i, x in enumerate(xs):
        bounce = 0.3 * abs(math.sin(2 * math.pi * t / 0.8 + i))
        items.append((SPHERE, (x, 0.6 + bounce, 0.0), (x0[i] - x) / 0.6))
    items.sort(key=lambda it: float(np.dot(it[1], VIEW)))
    parts = [instance_faces(*it) for it in items]
    obj = [np.concatenate([p[k] for p in parts]) for k in range(4)]
    gr = instance_faces(GROUND, (0, 0, 0), 0.0)

    def group(g):
        return (project(g[0]), g[0], g[1], g[2], g[3])

    return group(gr), group(tuple(obj))


# ---------------------------------------------------------------- iluminacao
def light(N, P, base):
    L = SUN - P
    L = L / np.linalg.norm(L, axis=-1, keepdims=True)
    NL = np.einsum("...i,...i->...", N, L)
    diff = np.maximum(NL, 0)
    R = 2 * NL[..., None] * N - L
    spec = np.maximum(R @ VIEW, 0) ** SHINE * (NL > 0)
    return np.clip(base * (KA + KD * diff)[..., None] + KS * spec[..., None], 0, 1).astype(np.float32)


def make_attrs(mode, PV, NV, FN, mat):
    base = MATS[mat]
    if mode == "flat":
        rgb = light(FN, PV.mean(axis=1), base)
        return np.repeat(rgb[:, None, :], 4, axis=1)
    if mode == "gouraud":
        return light(NV, PV, base[:, None, :])
    return np.concatenate([NV, PV], axis=-1).astype(np.float32)


def shade(mode, attrs, mat_pix):
    if mode == "phong":
        N = attrs[:, :3]
        N = N / np.linalg.norm(N, axis=1, keepdims=True)
        rgb = light(N, attrs[:, 3:], MATS[mat_pix])
    else:
        rgb = np.clip(attrs, 0, 1)
    return (rgb * 255).astype(np.uint8)


# ---------------------------------------------------------------- scanline
def make_spans(P, A, check=False):
    """Interseccoes aresta x linha de varredura (intervalo [ymin, ymax)), ordena por x e
    forma pares (esquerda, direita). Atributos interpolados linearmente ao longo das arestas."""
    k = A.shape[-1]
    P2 = np.roll(P, -1, axis=1)
    A2 = np.roll(A, -1, axis=1)
    x1 = P[..., 0].ravel()
    y1 = P[..., 1].ravel()
    x2 = P2[..., 0].ravel()
    y2 = P2[..., 1].ravel()
    r0 = np.maximum(np.ceil(np.minimum(y1, y2)).astype(np.int64), 0)
    r1 = np.minimum(np.ceil(np.maximum(y1, y2)).astype(np.int64), H)
    cnt = np.maximum(r1 - r0, 0)
    total = int(cnt.sum())
    e = np.repeat(np.arange(cnt.size), cnt)
    starts = np.cumsum(cnt) - cnt
    rows = r0[e] + (np.arange(total) - starts[e])
    t = (rows - y1[e]) / (y2[e] - y1[e])
    x = x1[e] + t * (x2[e] - x1[e])
    a1 = A.reshape(-1, k)[e]
    a = a1 + t[:, None].astype(np.float32) * (A2.reshape(-1, k)[e] - a1)
    poly = e // 4
    key = poly * H + rows
    order = np.lexsort((x, key))
    key, x, a, poly, rows = key[order], x[order], a[order], poly[order], rows[order]
    if check:
        assert total % 2 == 0 and np.array_equal(key[0::2], key[1::2]), "interseccoes nao vieram em pares"
    xL, xR = x[0::2], x[1::2]
    x0 = np.maximum(np.ceil(xL), 0).astype(np.int64)
    x1c = np.minimum(np.ceil(xR), W).astype(np.int64)
    return rows[0::2], poly[0::2], xL, xR, a[0::2], a[1::2], x0, x1c


def expand(sp, mode):
    rows, poly, xL, xR, aL, aR, x0, x1 = sp
    cnt = np.maximum(x1 - x0, 0)
    tot = int(cnt.sum())
    si = np.repeat(np.arange(cnt.size), cnt)
    px = x0[si] + (np.arange(tot) - (np.cumsum(cnt) - cnt)[si])
    if mode == "flat":
        at = aL[si]
    else:
        tt = ((px - xL[si]) / (xR - xL)[si]).astype(np.float32)
        at = aL[si] + tt[:, None] * (aR - aL)[si]
    return px, rows[si], poly[si], at


def draw_C(fb, grp, mode, T, check):
    P, PV, NV, FN, mat = grp
    t0 = time.perf_counter()
    A = make_attrs(mode, PV, NV, FN, mat)
    t1 = time.perf_counter()
    sp = make_spans(P, A, check)
    t2 = time.perf_counter()
    px, py, pp, at = expand(sp, mode)
    t3 = time.perf_counter()
    col = shade(mode, at, mat[pp])
    t4 = time.perf_counter()
    fb[px, py] = col
    t5 = time.perf_counter()
    for name, d in (("atributos", t1 - t0), ("varredura", t2 - t1), ("expandir", t3 - t2), ("sombrear", t4 - t3), ("escrever", t5 - t4)):
        T[name] = T.get(name, 0.0) + d
    return int(px.size)


def draw_B(fb, grp, mode, T, check):
    P, PV, NV, FN, mat = grp
    n = 0
    for i in range(P.shape[0]):
        s = slice(i, i + 1)
        A = make_attrs(mode, PV[s], NV[s], FN[s], mat[s])
        px, py, pp, at = expand(make_spans(P[s], A, check), mode)
        fb[px, py] = shade(mode, at, mat[s][pp])
        n += px.size
    return n


def draw_A(fb, grp, mode, T, check):
    P, PV, NV, FN, mat = grp
    A = make_attrs(mode, PV, NV, FN, mat)
    rows, poly, xL, xR, aL, aR, x0, x1 = make_spans(P, A, check)
    n = 0
    for i in range(rows.size):
        a, b = int(x0[i]), int(x1[i])
        if b <= a:
            continue
        if mode == "flat":
            fb[a:b, rows[i]] = (np.clip(aL[i], 0, 1) * 255).astype(np.uint8)
        else:
            xs = np.arange(a, b)
            tt = ((xs - xL[i]) / (xR[i] - xL[i])).astype(np.float32)
            at = aL[i] + tt[:, None] * (aR[i] - aL[i])
            fb[a:b, rows[i]] = shade(mode, at, np.full(b - a, mat[poly[i]], np.int32))
        n += b - a
    return n


DRAW = {"A": draw_A, "B": draw_B, "C": draw_C}


def render(fb, t, mode, strat, ground_mode, T, check=False):
    t0 = time.perf_counter()
    ground, objs = build_scene(t)
    T["geometria"] = T.get("geometria", 0.0) + time.perf_counter() - t0
    fb[:] = SKY
    npx = DRAW[strat](fb, ground, ground_mode, T, check)
    npx += DRAW[strat](fb, objs, mode, T, check)
    return npx, objs[0].shape[0]


def bench(screen, fb, mode, strat, ground_mode, nframes):
    T, ms = {}, []
    for f in range(nframes + 3):
        t0 = time.perf_counter()
        render(fb, f / 60.0, mode, strat, ground_mode, T)
        t1 = time.perf_counter()
        pygame.surfarray.blit_array(screen, fb)
        t2 = time.perf_counter()
        if f >= 3:
            ms.append((t2 - t0) * 1000)
            T["blit"] = T.get("blit", 0.0) + (t2 - t1)
        if f == 2:
            T.clear()
    return np.array(ms), {k: v * 1000 / nframes for k, v in T.items()}


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else None
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    fb = np.zeros((W, H, 3), np.uint8)

    # 1) corretude: as tres estrategias precisam produzir a mesma imagem
    imgs = {}
    for mode in ("flat", "gouraud", "phong"):
        for strat in "ABC":
            render(fb, 0.5, mode, strat, mode, {}, check=True)
            imgs[(mode, strat)] = fb.copy()
        diff_ab = int(np.abs(imgs[(mode, "A")].astype(int) - imgs[(mode, "B")]).max())
        diff_bc = int(np.abs(imgs[(mode, "B")].astype(int) - imgs[(mode, "C")]).max())
        print(f"[corretude] {mode:8s} dif max A-B={diff_ab} B-C={diff_bc}")
        if outdir:
            pygame.surfarray.blit_array(screen, imgs[(mode, "C")])
            pygame.image.save(screen, os.path.join(outdir, f"cena_{mode}.png"))

    npx, nfaces = render(fb, 0.5, "phong", "C", "phong", {})
    print(f"[cena] {nfaces} faces visiveis fora o terreno, {npx} pixels escritos (com sobreposicao)\n")

    # 2) desempenho
    print(f"{'modo':8s} {'terreno':8s} {'estrat':6s} {'ms/quadro':>10s} {'p95':>8s} {'FPS':>7s}")
    runs = []
    for mode in ("flat", "gouraud", "phong"):
        for strat, n in (("A", 15), ("B", 40), ("C", 200)):
            runs.append((mode, mode, strat, n))
    runs.append(("phong", "flat", "C", 200))
    runs.append(("phong", "flat", "B", 40))
    detail = {}
    for mode, gmode, strat, n in runs:
        ms, T = bench(screen, fb, mode, strat, gmode, n)
        detail[(mode, gmode, strat)] = T
        print(f"{mode:8s} {gmode:8s} {strat:6s} {ms.mean():10.2f} {np.percentile(ms, 95):8.2f} {1000 / ms.mean():7.1f}")

    print("\n[detalhe estrategia C, ms/quadro]")
    for key, T in detail.items():
        if key[2] == "C":
            print(f"  {key[0]:8s} terreno={key[1]:8s} " + "  ".join(f"{k}={v:.2f}" for k, v in T.items()))
    pygame.quit()


if __name__ == "__main__":
    main()
