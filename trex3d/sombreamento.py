"""Sombreamento de poligonos (apostila Un. 8, secao 8.2): constante, Gouraud e Phong.

- constante: uma cor por face, calculada no centro da face;
- gouraud: iluminacao nos vertices e interpolacao linear das cores (8.2.2);
- phong: interpolacao da normal e da posicao, e iluminacao em cada pixel (8.2.3).
As faces de uma chamada seguem a ordem da lista (ordem do pintor): o chamador junta os objetos do
mais longe ao mais perto.
"""
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from trex3d import luz as luz_mod
from trex3d import mat, modelos, raster, visibilidade

MODOS = ("constante", "gouraud", "phong")


@dataclass
class Faces:
    """Faces visiveis prontas para pintar (N faces de 4 vertices; triangulos repetem o ultimo)."""
    P: np.ndarray      # (N, 4, 2) vertices em pixels
    PV: np.ndarray     # (N, 4, 3) vertices no mundo
    NV: np.ndarray     # (N, 4, 3) normais unitarias dos vertices, no mundo
    FN: np.ndarray     # (N, 3)    normais unitarias das faces, no mundo
    ids: np.ndarray    # (N,)      indice do material de cada face


def preparar_faces(malha, m_modelo, camera, id_material):
    """Leva a malha ao mundo, descarta as faces de costas e projeta as demais na tela.

    A matriz de modelo so pode ter rotacao, translacao e escala uniforme (as normais dos vertices
    giram com a parte 3x3 e sao renormalizadas).
    """
    if not mat.so_rotacao_e_escala_uniforme(m_modelo):
        raise ValueError("escala nao uniforme: as normais dos vertices ficariam incorretas")
    m3 = np.asarray(m_modelo, dtype=float)[:3, :3]
    vertices = mat.aplicar(m_modelo, malha.vertices)
    normais_v = malha.normais_vertice @ m3.T
    normais_v = normais_v / np.linalg.norm(normais_v, axis=1, keepdims=True)
    normais_f = modelos.normais_das_faces(vertices[malha.faces])
    visiveis = np.flatnonzero(visibilidade.faces_voltadas_para_o_observador(normais_f, camera.n))
    idx = malha.faces[visiveis]
    pv = vertices[idx]
    p = camera.projetar(pv.reshape(-1, 3))[:, :2].reshape(-1, 4, 2)
    return Faces(P=p, PV=pv, NV=normais_v[idx], FN=normais_f[visiveis],
                 ids=np.full(len(visiveis), id_material, dtype=np.int64))


def juntar(lista):
    if not lista:
        return Faces(np.zeros((0, 4, 2)), np.zeros((0, 4, 3)), np.zeros((0, 4, 3)),
                     np.zeros((0, 3)), np.zeros(0, dtype=np.int64))
    return Faces(*(np.concatenate([getattr(f, campo) for f in lista]) for campo in ("P", "PV", "NV", "FN", "ids")))


class _Tabela:
    """Coeficientes dos materiais empilhados, para indexar por face, vertice ou pixel."""

    def __init__(self, materiais):
        self.ka = np.stack([m.ka for m in materiais])
        self.kd = np.stack([m.kd for m in materiais])
        self.ks = np.stack([m.ks for m in materiais])
        self.emissao = np.stack([m.emissao for m in materiais])
        self.n = np.array([m.n for m in materiais], dtype=float)

    def por(self, ids):
        return SimpleNamespace(ka=self.ka[ids], kd=self.kd[ids], ks=self.ks[ids],
                               emissao=self.emissao[ids], n=self.n[ids][..., None])


def _centros(pv):
    """Media dos vertices distintos de cada face (triangulo: 3 vertices)."""
    quadrilatero = ~np.all(pv[:, 3] == pv[:, 2], axis=1)[:, None]
    soma = pv[:, 0] + pv[:, 1] + pv[:, 2] + np.where(quadrilatero, pv[:, 3], 0.0)
    return soma / np.where(quadrilatero, 4.0, 3.0)


def calcular_cores(modo, faces, materiais, luz, direcao_do_observador, largura, altura):
    """Devolve (px, py, face, rgb) com rgb em [0, 1] para cada pixel coberto."""
    if modo not in MODOS:
        raise ValueError(f"modo de sombreamento desconhecido: {modo!r} (use {MODOS})")
    tabela = _Tabela(materiais)
    s = direcao_do_observador

    if modo == "constante":
        cor = luz_mod.iluminar(_centros(faces.PV), faces.FN, tabela.por(faces.ids), luz, s)
        px, py, poly, _ = raster.rasterizar(faces.P, None, largura, altura)
        return px, py, poly, cor[poly]

    if modo == "gouraud":
        nos = luz_mod.iluminar(faces.PV, faces.NV, tabela.por(faces.ids[:, None]), luz, s)
        return raster.rasterizar(faces.P, nos, largura, altura)

    atributos = np.concatenate([faces.NV, faces.PV], axis=-1)
    px, py, poly, a = raster.rasterizar(faces.P, atributos, largura, altura)
    normal = a[:, :3]
    normal = normal / np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
    return px, py, poly, luz_mod.iluminar(a[:, 3:], normal, tabela.por(faces.ids[poly]), luz, s)


def pintar_faces(fb, modo, faces, materiais, luz, direcao_do_observador):
    """Pinta as faces no framebuffer (largura, altura, 3) uint8. Devolve o numero de pixels escritos."""
    px, py, _, rgb = calcular_cores(modo, faces, materiais, luz, direcao_do_observador,
                                    fb.shape[0], fb.shape[1])
    fb[px, py] = luz_mod.para_uint8(rgb)
    return int(px.size)
