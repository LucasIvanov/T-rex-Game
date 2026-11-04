"""Testes dos modelos (parte 3 do plano): cubo, esfera por faixas e fatias, plano.

Valores esperados calculados a mao. Convencoes: faces dadas em sentido anti-horario visto de fora
(normal para fora), triangulos completados com o ultimo vertice repetido, objetos centrados na origem.
Conferir as contas e acrescentar casos proprios em CASOS_DO_USUARIO.
"""
import math

import numpy as np
import pytest

from trex3d import modelos

R3 = math.sqrt(3)


def aprox(v, tol=1e-9):
    return pytest.approx(v, abs=tol)


def arestas(malha):
    """Conjunto de arestas (par de indices ordenado) ignorando o vertice repetido dos triangulos."""
    res = []
    for f in malha.faces:
        ids = [f[0], f[1], f[2]] + ([f[3]] if f[3] != f[2] else [])
        for a, b in zip(ids, ids[1:] + ids[:1]):
            res.append(tuple(sorted((int(a), int(b)))))
    return res


def test_normal_quadrado_anti_horario():
    # soma dos produtos vetoriais (0,0,2) -> (0,0,1)
    quad = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
    assert modelos.normal_da_face(quad) == aprox([0, 0, 1])


def test_normal_quadrado_horario_inverte():
    quad = [(0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)]
    assert modelos.normal_da_face(quad) == aprox([0, 0, -1])


def test_normal_triangulo_com_vertice_repetido():
    tri = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 1, 0)]
    assert modelos.normal_da_face(tri) == aprox([0, 0, 1])


def test_normal_nao_depende_da_origem():
    quad = np.array([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)], float) + (5, -3, 7)
    assert modelos.normal_da_face(quad) == aprox([0, 0, 1])


def test_face_degenerada_e_erro():
    with pytest.raises(ValueError):
        modelos.normal_da_face([(0, 0, 0), (1, 0, 0), (2, 0, 0), (2, 0, 0)])


def test_cubo_contagens():
    c = modelos.criar_cubo(2)
    assert c.vertices.shape == (24, 3)   # 6 faces x 4 vertices proprios (arestas vivas)
    assert c.faces.shape == (6, 4)
    assert c.normais_face.shape == (6, 3)


def test_cubo_vertices_nos_cantos():
    c = modelos.criar_cubo(2)
    assert set(np.unique(np.abs(c.vertices))) == {1.0}
    cantos = {tuple(v) for v in c.vertices}
    assert len(cantos) == 8


def test_cubo_normais_das_faces_sao_os_6_eixos_para_fora():
    c = modelos.criar_cubo(2)
    achadas = {tuple(int(round(x)) for x in n) for n in c.normais_face}
    assert achadas == {(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)}
    # centro de cada face = normal * (lado/2) = normal * 1
    assert c.centros_das_faces == aprox(c.normais_face)


def test_cubo_normal_do_vertice_e_a_da_sua_face():
    c = modelos.criar_cubo(2)
    for i, face in enumerate(c.faces):
        for k in face:
            assert c.normais_vertice[k] == aprox(c.normais_face[i])


def test_cubo_lado_muda_o_tamanho():
    assert np.abs(modelos.criar_cubo(1.2).vertices).max() == aprox(0.6)


def test_esfera_octaedro():
    # 2 faixas x 4 fatias: polos (0,+-1,0) e equador (1,0,0),(0,0,1),(-1,0,0),(0,0,-1)
    e = modelos.criar_esfera(1, 2, 4)
    assert e.vertices.shape == (6, 3) and e.faces.shape == (8, 4)
    achadas = {tuple(int(round(x * R3)) for x in n) for n in e.normais_face}
    esperadas = {(sx, sy, sz) for sx in (1, -1) for sy in (1, -1) for sz in (1, -1)}
    assert achadas == esperadas   # normais (+-1,+-1,+-1)/raiz(3), uma de cada
    # primeira face: calota superior, entre os meridianos 0 e 1 -> face dos eixos +x, +y, +z
    assert e.normais_face[0] == aprox([1 / R3, 1 / R3, 1 / R3])


def test_esfera_contagens_6x8():
    e = modelos.criar_esfera(2, 6, 8)
    assert e.vertices.shape == (2 + 5 * 8, 3)            # 2 polos + 5 aneis de 8
    assert e.faces.shape == (6 * 8, 4)                   # 8 + 4*8 + 8
    triangulos = int(np.sum(e.faces[:, 3] == e.faces[:, 2]))
    assert triangulos == 16


def test_esfera_vertices_na_superficie_e_um_conhecido():
    e = modelos.criar_esfera(2, 6, 8)
    assert np.linalg.norm(e.vertices, axis=1) == aprox(np.full(42, 2.0))
    # primeiro anel (faixa 1, 30 graus a partir do polo), meridiano 0: (2 sen30, 2 cos30, 0) = (1, raiz3, 0)
    assert any(np.allclose(v, (1, R3, 0), atol=1e-9) for v in e.vertices)
    assert e.vertices[0] == aprox([0, 2, 0])
    assert e.vertices[-1] == aprox([0, -2, 0])


def test_esfera_normal_do_vertice_e_radial_unitaria():
    e = modelos.criar_esfera(2, 6, 8)
    assert e.normais_vertice == aprox(e.vertices / 2)


@pytest.mark.parametrize("faixas, fatias", [(2, 3), (2, 4), (6, 8), (10, 16)])
def test_esfera_faces_para_fora_e_unitarias(faixas, fatias):
    e = modelos.criar_esfera(1.5, faixas, fatias)
    assert np.linalg.norm(e.normais_face, axis=1) == aprox(np.ones(e.faces.shape[0]))
    assert np.sum(e.normais_face * e.centros_das_faces, axis=1).min() > 0


@pytest.mark.parametrize("faixas, fatias", [(2, 3), (2, 4), (6, 8), (10, 16)])
def test_esfera_e_superficie_fechada_euler_2(faixas, fatias):
    e = modelos.criar_esfera(1, faixas, fatias)
    lista = arestas(e)
    unicas = set(lista)
    assert all(lista.count(a) == 2 for a in unicas)      # toda aresta e compartilhada por exatamente 2 faces
    assert e.vertices.shape[0] - len(unicas) + e.faces.shape[0] == 2


def test_esfera_soma_das_normais_de_face_e_zero():
    e = modelos.criar_esfera(1, 6, 8)   # simetria rotacional em y e espelho em y
    assert e.normais_face.sum(axis=0) == aprox([0, 0, 0])


def test_esfera_parametros_invalidos():
    with pytest.raises(ValueError):
        modelos.criar_esfera(1, 1, 8)
    with pytest.raises(ValueError):
        modelos.criar_esfera(1, 6, 2)
    with pytest.raises(ValueError):
        modelos.criar_esfera(0, 6, 8)


def test_plano():
    p = modelos.criar_plano(-8, 8, -6, 8, y=0)
    assert p.vertices.shape == (4, 3) and p.faces.shape == (1, 4)
    assert {tuple(v) for v in p.vertices} == {(-8, 0, -6), (8, 0, -6), (8, 0, 8), (-8, 0, 8)}
    assert p.normais_face[0] == aprox([0, 1, 0])
    assert p.normais_vertice == aprox(np.tile([0, 1, 0], (4, 1)))
    assert p.centros_das_faces[0] == aprox([0, 0, 1])    # centro: x = 0, z = (-6 + 8) / 2


def test_plano_na_altura_y():
    p = modelos.criar_plano(0, 1, 0, 1, y=2.5)
    assert np.all(p.vertices[:, 1] == 2.5)


def test_centro_de_triangulo_nao_conta_o_vertice_repetido():
    m = modelos.Malha([(0, 0, 0), (3, 0, 0), (0, 3, 0)], [(0, 0, 1)] * 3, [(0, 1, 2, 2)])
    assert m.centros_das_faces[0] == aprox([1, 1, 0])    # media de 3 vertices, nao de 4


# Acrescente 2 a 3 casos seus: (nome da funcao, argumentos, numero de vertices, numero de faces).
CASOS_DO_USUARIO = []


@pytest.mark.parametrize("fabrica, args, n_vertices, n_faces", CASOS_DO_USUARIO)
def test_casos_do_usuario(fabrica, args, n_vertices, n_faces):
    m = getattr(modelos, fabrica)(*args)
    assert m.vertices.shape[0] == n_vertices and m.faces.shape[0] == n_faces
