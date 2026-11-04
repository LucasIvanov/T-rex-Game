"""Testes do nucleo matematico (parte 1 do plano).

Os valores esperados foram calculados a mao a partir das definicoes da apostila
(vetor coluna, P' = M * P, rotacao positiva = anti-horaria, mao direita) e NAO saem do codigo.
Conferir cada conta e acrescentar casos proprios em CASOS_DO_USUARIO.
"""
import math

import numpy as np
import pytest

from trex3d import mat

R90 = math.pi / 2


def aprox(v):
    return pytest.approx(v, abs=1e-9)


def test_escalar():
    # 1*4 + 2*5 + 3*6 = 32
    assert mat.escalar((1, 2, 3), (4, 5, 6)) == aprox(32)


def test_vetorial_eixos():
    # x vezes y = z (mao direita)
    assert mat.vetorial((1, 0, 0), (0, 1, 0)) == aprox([0, 0, 1])


def test_vetorial_geral():
    # (2*6-3*5, 3*4-1*6, 1*5-2*4) = (-3, 6, -3)
    assert mat.vetorial((1, 2, 3), (4, 5, 6)) == aprox([-3, 6, -3])


def test_vetorial_anticomutativo():
    a, b = (1, 2, 3), (4, 5, 6)
    assert mat.vetorial(a, b) == aprox(-mat.vetorial(b, a))


def test_normalizar():
    # |(3,0,4)| = 5
    assert mat.normalizar((3, 0, 4)) == aprox([0.6, 0, 0.8])


def test_normalizar_vetor_nulo():
    with pytest.raises(ValueError):
        mat.normalizar((0, 0, 0))


def test_translacao():
    # (1,1,1) + (2,-3,5) = (3,-2,6)
    assert mat.aplicar(mat.translacao(2, -3, 5), (1, 1, 1)) == aprox([3, -2, 6])


def test_escala():
    # (1,1,1) com (2,3,4) = (2,3,4); a origem nao se move
    assert mat.aplicar(mat.escala(2, 3, 4), (1, 1, 1)) == aprox([2, 3, 4])
    assert mat.aplicar(mat.escala(2, 3, 4), (0, 0, 0)) == aprox([0, 0, 0])


def test_rotacao_z_90():
    # x -> y e y -> -x (anti-horario visto de +z)
    assert mat.aplicar(mat.rot_z(R90), (1, 0, 0)) == aprox([0, 1, 0])
    assert mat.aplicar(mat.rot_z(R90), (0, 1, 0)) == aprox([-1, 0, 0])


def test_rotacao_x_90():
    # tabela da apostila: em torno de x, y vai para z
    assert mat.aplicar(mat.rot_x(R90), (0, 1, 0)) == aprox([0, 0, 1])


def test_rotacao_y_90():
    # tabela da apostila: em torno de y, z vai para x
    assert mat.aplicar(mat.rot_y(R90), (0, 0, 1)) == aprox([1, 0, 0])


def test_rotacao_180_em_z():
    assert mat.aplicar(mat.rot_z(math.pi), (1, 2, 0)) == aprox([-1, -2, 0])


def test_compor_ordem_importa():
    # compor(A, B) aplica B primeiro.
    # escala(2,1,1) e depois translacao(1,0,0) em (1,0,0): 1 -> 2 -> 3
    depois_translada = mat.compor(mat.translacao(1, 0, 0), mat.escala(2, 1, 1))
    assert mat.aplicar(depois_translada, (1, 0, 0)) == aprox([3, 0, 0])
    # translacao(1,0,0) e depois escala(2,1,1): 1 -> 2 -> 4
    depois_escala = mat.compor(mat.escala(2, 1, 1), mat.translacao(1, 0, 0))
    assert mat.aplicar(depois_escala, (1, 0, 0)) == aprox([4, 0, 0])


def test_compor_sem_argumentos_e_identidade():
    assert mat.compor() == aprox(np.eye(4))


def test_rotacao_em_torno_de_ponto():
    # ponto (2,1,0) em torno de (1,1,0), 90 graus em z:
    # vetor relativo (1,0,0) -> (0,1,0); soma o centro -> (1,2,0)
    m = mat.em_torno_de(mat.rot_z(R90), (1, 1, 0))
    assert mat.aplicar(m, (2, 1, 0)) == aprox([1, 2, 0])
    # o proprio centro nao se move
    assert mat.aplicar(m, (1, 1, 0)) == aprox([1, 1, 0])


@pytest.mark.parametrize(
    "m, inversa",
    [
        (mat.translacao(2, -3, 5), mat.translacao(-2, 3, -5)),
        (mat.escala(2, 4, 0.5), mat.escala(0.5, 0.25, 2)),
        (mat.rot_x(0.7), mat.rot_x(-0.7)),
        (mat.rot_y(1.3), mat.rot_y(-1.3)),
        (mat.rot_z(-2.1), mat.rot_z(2.1)),
    ],
)
def test_produto_pela_inversa_e_identidade(m, inversa):
    assert m @ inversa == aprox(np.eye(4))


@pytest.mark.parametrize("rot", [mat.rot_x, mat.rot_y, mat.rot_z])
def test_rotacao_e_ortogonal_com_determinante_1(rot):
    r = rot(0.9)[:3, :3]
    assert r.T @ r == aprox(np.eye(3))
    assert np.linalg.det(r) == aprox(1.0)


def test_so_rotacao_e_escala_uniforme():
    assert mat.so_rotacao_e_escala_uniforme(mat.rot_x(0.4))
    assert mat.so_rotacao_e_escala_uniforme(mat.compor(mat.translacao(3, 2, 1), mat.rot_y(1.1), mat.escala(2, 2, 2)))
    assert not mat.so_rotacao_e_escala_uniforme(mat.escala(2, 1, 1))              # escala diferente por eixo
    assert not mat.so_rotacao_e_escala_uniforme(mat.compor(mat.rot_z(0.3), mat.escala(1, 3, 1)))


def test_aplicar_divide_pelo_fator_homogeneo():
    # w = 2: (2,4,6,2) -> (1,2,3)
    m = np.diag([1.0, 1.0, 1.0, 2.0])
    assert mat.aplicar(m, (2, 4, 6)) == aprox([1, 2, 3])


def test_aplicar_varios_pontos():
    pts = np.array([[1, 0, 0], [0, 1, 0]], float)
    esperado = np.array([[0, 1, 0], [-1, 0, 0]], float)
    assert mat.aplicar(mat.rot_z(R90), pts) == aprox(esperado)


def test_aplicar_w_zero():
    m = np.eye(4)
    m[3, 3] = 0.0
    with pytest.raises(ValueError):
        mat.aplicar(m, (1, 2, 3))


# Acrescente aqui 2 a 3 casos seus: (matriz montada com as funcoes, ponto, resultado que VOCE calculou).
CASOS_DO_USUARIO = []


@pytest.mark.parametrize("m, ponto, esperado", CASOS_DO_USUARIO)
def test_casos_do_usuario(m, ponto, esperado):
    assert mat.aplicar(m, ponto) == aprox(esperado)
