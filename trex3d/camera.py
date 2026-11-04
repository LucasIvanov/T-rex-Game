"""Camera ortografica e janela->viewport (apostila Un. 5, secoes 5.2, 5.5 e 5.7.1).

Pipeline: mundo (SRU) -> observador (SRC) -> projecao ortografica -> janela->viewport (pixels).
n aponta da cena para o olho (N = VRP - P). A profundidade devolvida e o z do SRC: quanto maior,
mais perto do observador. O z nao entra em x e y, mas e mantido para a ocultacao (apostila 5.4).
"""
import numpy as np

from trex3d import mat


def base_do_observador(n, view_up):
    """Vetores unitarios (u, v, n) do SRC. V = Y - (Y.n) n; u = v x n (mao direita)."""
    n_unit = mat.normalizar(n)
    y = np.asarray(view_up, dtype=float)
    v = y - mat.escalar(y, n_unit) * n_unit
    if np.allclose(v, 0.0, atol=1e-12 * max(1.0, mat.escalar(y, y))):
        raise ValueError("view up paralelo a N: o plano de observacao fica sem direcao 'para cima'")
    v = mat.normalizar(v)
    u = mat.normalizar(mat.vetorial(v, n_unit))
    return u, v, n_unit


def matriz_sru_para_src(vrp, n, view_up):
    """R * T(-VRP): leva o VRP para a origem e alinha u, v, n com x, y, z."""
    u, v, n_unit = base_do_observador(n, view_up)
    r = np.eye(4)
    r[0, :3], r[1, :3], r[2, :3] = u, v, n_unit
    return mat.compor(r, mat.translacao(-vrp[0], -vrp[1], -vrp[2]))


def matriz_janela_viewport(janela, viewport, y_invertido=True):
    """janela = (xmin, ymin, xmax, ymax); viewport = (umin, vmin, umax, vmax).

    y_invertido: T(umin, vmax) E_Ox S T(-xmin, -ymin); caso contrario T(umin, vmin) S T(-xmin, -ymin).
    O z passa sem alteracao.
    """
    xmin, ymin, xmax, ymax = janela
    umin, vmin, umax, vmax = viewport
    if xmax == xmin or ymax == ymin:
        raise ValueError("janela de largura ou altura nula")
    sx = (umax - umin) / (xmax - xmin)
    sy = (vmax - vmin) / (ymax - ymin)
    if y_invertido:
        return mat.compor(
            mat.translacao(umin, vmax, 0),
            mat.escala(1, -1, 1),
            mat.escala(sx, sy, 1),
            mat.translacao(-xmin, -ymin, 0),
        )
    return mat.compor(mat.translacao(umin, vmin, 0), mat.escala(sx, sy, 1), mat.translacao(-xmin, -ymin, 0))


class CameraOrtografica:
    def __init__(self, vrp, n, view_up, janela, viewport, y_invertido=True):
        self.u, self.v, self.n = base_do_observador(n, view_up)
        self.vrp = np.asarray(vrp, dtype=float)
        self.m_sru_src = matriz_sru_para_src(vrp, n, view_up)
        self.m_jp = matriz_janela_viewport(janela, viewport, y_invertido)
        self.m_total = mat.compor(self.m_jp, self.m_sru_src)

    def para_observador(self, pontos):
        """Pontos do mundo (3,) ou (N, 3) -> coordenadas do SRC."""
        return mat.aplicar(self.m_sru_src, pontos)

    def projetar(self, pontos):
        """Pontos do mundo (3,) ou (N, 3) -> (px, py, profundidade)."""
        return mat.aplicar(self.m_total, pontos)
