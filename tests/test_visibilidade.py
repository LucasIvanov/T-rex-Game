"""Testes de visibilidade (parte 4 do plano): faces de costas pela normal e ordem do pintor.

Valores esperados calculados a mao. Regra (apostila 5.9.3): uma face e visivel se
D = N.O + d > 0 (d = -N.P0); na projecao ortografica o observador esta no infinito na direcao n
e a regra vira N.n > 0. Faces de perfil (N.n = 0) nao aparecem (area nula na tela).
Conferir as contas e acrescentar casos proprios em CASOS_DO_USUARIO.
"""
import math

import numpy as np
import pytest

from trex3d import camera, mat, modelos, visibilidade

R3_2 = math.sqrt(3) / 2
JANELA = (-4, -3, 4, 3)
VIEWPORT = (0, 0, 800, 600)


def aprox(v, tol=1e-9):
    return pytest.approx(v, abs=tol)


def cam(n, vrp=(0, 0, 0)):
    return camera.CameraOrtografica(vrp, n, (0, 1, 0), JANELA, VIEWPORT)


def normais_visiveis(malha, m, c):
    idx = visibilidade.faces_visiveis(malha, m, c)
    return {tuple(int(round(x)) for x in malha.normais_face[i]) for i in idx}


def test_lado_do_plano_observador_na_frente():
    # plano z = 2 (normal +z, ponto (0,0,2), d = -2), observador (1,2,5): D = 5 - 2 = 3
    assert visibilidade.lado_do_plano((0, 0, 1), (0, 0, 2), (1, 2, 5)) == aprox(3)


def test_lado_do_plano_atras_e_no_plano():
    assert visibilidade.lado_do_plano((0, 0, 1), (0, 0, 2), (0, 0, -1)) == aprox(-3)   # -1 - 2
    assert visibilidade.lado_do_plano((0, 0, 1), (0, 0, 2), (7, -3, 2)) == aprox(0)    # observador no plano


def test_lado_do_plano_varias_faces():
    d = visibilidade.lado_do_plano(np.array([[0, 0, 1], [0, 0, -1]]), np.array([[0, 0, 2], [0, 0, 2]]), (0, 0, 5))
    # face 1: (+1)*(5-2) = 3; face 2 (normal -z): (-1)*(5-2) = -3
    assert d == aprox([3, -3])


def test_cubo_vista_frontal_so_a_face_da_frente():
    # n = +z: so (0,0,1) tem N.n = 1; as de perfil (N.n = 0) saem
    c = modelos.criar_cubo(2)
    assert normais_visiveis(c, np.eye(4), cam((0, 0, 1))) == {(0, 0, 1)}


def test_cubo_elevacao_30_graus_frente_e_topo():
    # n = (0, 0.5, raiz3/2): +z -> 0.866, +y -> 0.5, -z -> -0.866, -y -> -0.5, +-x -> 0 (perfil)
    c = modelos.criar_cubo(2)
    assert normais_visiveis(c, np.eye(4), cam((0, 0.5, R3_2))) == {(0, 0, 1), (0, 1, 0)}


def test_cubo_girado_45_em_y_mostra_3_faces():
    # Ry(45): +z -> (.707,0,.707), -x -> (-.707,0,.707), +x -> (.707,0,-.707); n.N: .612, .612, -.612; topo .5
    c = modelos.criar_cubo(2)
    m = mat.rot_y(math.pi / 4)
    idx = visibilidade.faces_visiveis(c, m, cam((0, 0.5, R3_2)))
    assert len(idx) == 3


def test_translacao_nao_muda_a_visibilidade_na_ortografica():
    c = modelos.criar_cubo(2)
    cm = cam((0, 0.5, R3_2))
    a = visibilidade.faces_visiveis(c, np.eye(4), cm)
    b = visibilidade.faces_visiveis(c, mat.translacao(50, -20, 33), cm)
    assert list(a) == list(b)


def test_octaedro_vista_frontal_4_das_8_faces():
    # normais (+-1,+-1,+-1)/raiz3; N.n > 0 so com z = +1
    e = modelos.criar_esfera(1, 2, 4)
    assert len(visibilidade.faces_visiveis(e, np.eye(4), cam((0, 0, 1)))) == 4


def test_esfera_6x8_vista_frontal_metade_das_faces():
    # por simetria, nenhum centro de face tem z = 0; 24 de 48 faces voltadas para o observador
    e = modelos.criar_esfera(1, 6, 8)
    assert len(visibilidade.faces_visiveis(e, np.eye(4), cam((0, 0, 1)))) == 24


def test_face_visivel_aponta_para_o_observador():
    e = modelos.criar_esfera(1, 6, 8)
    cm = cam((0, 0.5, R3_2))
    idx = visibilidade.faces_visiveis(e, np.eye(4), cm)
    assert np.all(e.normais_face[idx] @ cm.n > 0)
    fora = np.setdiff1d(np.arange(e.faces.shape[0]), idx)
    assert np.all(e.normais_face[fora] @ cm.n <= 1e-9)


def test_mascara_direta_com_tolerancia():
    normais = np.array([[0, 0, 1], [0, 0, -1], [1, 0, 0], [0, 0.001, 1e-12]], float)
    m = visibilidade.faces_voltadas_para_o_observador(normais, (0, 0, 1))
    assert list(m) == [True, False, False, False]   # perfil e quase-perfil ficam de fora


def test_ordem_do_pintor_vista_frontal():
    # profundidade = z (n = +z): centros em z = 3, -2, 0  ->  do mais longe ao mais perto: indices 1, 2, 0
    centros = np.array([[0, 0, 3], [0, 0, -2], [0, 0, 0]], float)
    assert list(visibilidade.ordem_do_pintor(centros, cam((0, 0, 1)))) == [1, 2, 0]


def test_ordem_do_pintor_com_elevacao():
    # profundidade = n.P: (0,0,0) -> 0; (0,1,0) -> 0.5; (0,0,1) -> 0.866  ->  ordem 0, 1, 2
    centros = np.array([[0, 0, 1], [0, 0, 0], [0, 1, 0]], float)
    assert list(visibilidade.ordem_do_pintor(centros, cam((0, 0.5, R3_2)))) == [1, 2, 0]


def test_ordem_do_pintor_empate_mantem_a_ordem_original():
    centros = np.array([[5, 0, 0], [-5, 0, 0], [0, 0, 0]], float)   # mesma profundidade em vista frontal
    assert list(visibilidade.ordem_do_pintor(centros, cam((0, 0, 1)))) == [0, 1, 2]


def test_profundidade_dos_centros():
    centros = np.array([[0, 0, 1], [0, 1, 0]], float)
    assert visibilidade.profundidade_dos_centros(centros, cam((0, 0.5, R3_2))) == aprox([R3_2, 0.5])


def test_centro_do_objeto():
    e = modelos.criar_esfera(1, 6, 8)
    m = mat.translacao(4, 1, -2)
    assert visibilidade.centro_do_objeto(e, m) == aprox([4, 1, -2])


# Acrescente 2 a 3 casos seus: (n do observador, normal da face, esta_visivel).
CASOS_DO_USUARIO = []


@pytest.mark.parametrize("n, normal, visivel", CASOS_DO_USUARIO)
def test_casos_do_usuario(n, normal, visivel):
    m = visibilidade.faces_voltadas_para_o_observador(np.array([normal], float), n)
    assert bool(m[0]) == visivel
