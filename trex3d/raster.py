"""Preenchimento de poligonos por scanline (apostila Un. 6, secao 6.9), vetorizado com numpy.

Por linha y inteira: interseccao com as arestas, ordenacao em x e preenchimento entre pares. Uma
aresta vale para ymin <= y < ymax e cada linha pinta de ceil(x_esq) ate ceil(x_dir) - 1; assim uma
aresta compartilhada nao deixa buraco nem e pintada duas vezes. Vale para poligonos concavos sem
buracos. O pixel (x, y) e amostrado na coordenada inteira; framebuffer (largura, altura, 3) uint8,
indexado fb[x, y]. Os poligonos de uma chamada seguem a ordem da lista (ordem do pintor).
"""
import numpy as np


def rasterizar(P, A, largura, altura, interpolar=True):
    """Pixels cobertos por N poligonos de K vertices.

    P: (N, K, 2) vertices em pixels (triangulos: repita o ultimo vertice). A: (N, K, k) atributos por
    vertice, ou None. Devolve (px, py, poligono, atributos), um elemento por pixel, ordenados por
    poligono e depois por linha; atributos e None se A for None.
    interpolar=True: atributos interpolados linearmente nas arestas (em y) e ao longo da linha (em x);
    False: usa o valor do vertice de inicio do trecho (atributo constante por poligono).
    """
    P = np.asarray(P, dtype=float)
    n, K = P.shape[0], P.shape[1]
    if A is not None:
        A = np.asarray(A)
        if not np.issubdtype(A.dtype, np.floating):
            A = A.astype(float)
        k = A.shape[-1]

    P2 = np.roll(P, -1, axis=1)
    x1, y1 = P[..., 0].ravel(), P[..., 1].ravel()
    x2, y2 = P2[..., 0].ravel(), P2[..., 1].ravel()

    r0 = np.maximum(np.ceil(np.minimum(y1, y2)).astype(np.int64), 0)
    r1 = np.minimum(np.ceil(np.maximum(y1, y2)).astype(np.int64), altura)
    cnt = np.maximum(r1 - r0, 0)
    total = int(cnt.sum())
    if total % 2:
        raise ValueError("numero impar de interseccoes: contorno aberto ou coordenadas invalidas")

    e = np.repeat(np.arange(cnt.size), cnt)
    rows = r0[e] + (np.arange(total) - (np.cumsum(cnt) - cnt)[e])
    t = (rows - y1[e]) / (y2[e] - y1[e])
    x = x1[e] + t * (x2[e] - x1[e])
    poly = e // K

    order = np.lexsort((x, poly * altura + rows))
    x, rows, poly, e, t = x[order], rows[order], poly[order], e[order], t[order]

    xL, xR = x[0::2], x[1::2]
    rows_s, poly_s = rows[0::2], poly[0::2]
    x0 = np.clip(np.ceil(xL), 0, largura).astype(np.int64)
    x1c = np.clip(np.ceil(xR), 0, largura).astype(np.int64)

    cnt_px = np.maximum(x1c - x0, 0)
    tot_px = int(cnt_px.sum())
    si = np.repeat(np.arange(cnt_px.size), cnt_px)
    px = x0[si] + (np.arange(tot_px) - (np.cumsum(cnt_px) - cnt_px)[si])
    py = rows_s[si]
    pol = poly_s[si]

    if A is None:
        return px, py, pol, None

    A_flat = A.reshape(-1, k)
    A_next = np.roll(A, -1, axis=1).reshape(-1, k)
    a_edge = A_flat[e] + t[:, None].astype(A.dtype) * (A_next[e] - A_flat[e])
    aL, aR = a_edge[0::2], a_edge[1::2]
    if not interpolar:
        return px, py, pol, aL[si]
    u = ((px - xL[si]) / (xR - xL)[si]).astype(A.dtype)
    return px, py, pol, aL[si] + u[:, None] * (aR - aL)[si]


def preencher(fb, P, cores):
    """Pinta N poligonos (N, K, 2) com uma cor uint8 (N, 3) cada, na ordem da lista. Devolve o
    numero de pixels escritos (com sobreposicao)."""
    px, py, poly, _ = rasterizar(P, None, fb.shape[0], fb.shape[1])
    fb[px, py] = np.asarray(cores)[poly]
    return int(px.size)
