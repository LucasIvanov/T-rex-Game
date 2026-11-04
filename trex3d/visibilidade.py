"""Visibilidade (apostila Un. 5, secao 5.9.3): faces de costas pela normal e ordem do pintor.

- Face visivel: D = N.O + d > 0, com d = -N.P0 (lado_do_plano). Na ortografica o observador esta no
  infinito na direcao n e a regra vira N.n > 0; faces de perfil (N.n = 0) nao aparecem.
- Correta sozinha so para solidos convexos fechados (cubo, esfera): sobra ordenar os objetos.
- Ordem do pintor: do objeto mais longe ao mais perto, pela profundidade do centro (z do SRC).
  O terreno nao entra: e desenhado antes de todos os objetos.
"""
import numpy as np

from trex3d import mat, modelos

TOLERANCIA = 1e-9


def lado_do_plano(normal, ponto_da_face, observador):
    """D = N.(O - P0) = N.O + d. D > 0: observador do lado da normal; D = 0: no plano."""
    n = np.asarray(normal, dtype=float)
    p0 = np.asarray(ponto_da_face, dtype=float)
    o = np.asarray(observador, dtype=float)
    return np.sum(n * (o - p0), axis=-1)


def faces_voltadas_para_o_observador(normais, direcao_do_observador, tolerancia=TOLERANCIA):
    """Mascara booleana (F,) para normais (F, 3): N.n > tolerancia."""
    return np.asarray(normais, dtype=float) @ np.asarray(direcao_do_observador, dtype=float) > tolerancia


def faces_visiveis(malha, m_modelo, camera):
    """Indices das faces de `malha`, depois de levada ao mundo por m_modelo, voltadas para a camera."""
    vertices_mundo = mat.aplicar(m_modelo, malha.vertices)
    normais = modelos.normais_das_faces(vertices_mundo[malha.faces])
    return np.flatnonzero(faces_voltadas_para_o_observador(normais, camera.n))


def centro_do_objeto(malha, m_modelo):
    """Media dos vertices da malha, levada ao mundo."""
    return mat.aplicar(m_modelo, malha.vertices.mean(axis=0))


def profundidade_dos_centros(centros, camera):
    """z do SRC de cada centro (N, 3): maior = mais perto do observador."""
    return camera.para_observador(np.asarray(centros, dtype=float))[..., 2]


def ordem_do_pintor(centros, camera):
    """Indices dos objetos do mais longe ao mais perto (desenhar nessa ordem); empate mantem a ordem."""
    return np.argsort(profundidade_dos_centros(centros, camera), kind="stable")
