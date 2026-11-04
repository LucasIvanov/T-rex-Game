"""Testes da camera ortografica e da janela->viewport (parte 2 do plano).

Valores esperados calculados a mao (nao saem do codigo). Camera a 30 graus de elevacao:
sen = 0,5 e cos = raiz(3)/2, entao
    n = (0, 0.5, raiz(3)/2)        direcao do observador (de fora da cena para o olho)
    v = (0, raiz(3)/2, -0.5)       "para cima" na tela (Y=(0,1,0) projetado no plano de observacao)
    u = v x n = (1, 0, 0)          "para a direita" na tela
Janela (-4,-3)-(4,3) no SRC -> viewport (0,0)-(800,600), y da tela para baixo: 100 px por unidade.
Conferir cada conta e acrescentar casos proprios em CASOS_DO_USUARIO.
"""
import math

import numpy as np
import pytest

from trex3d import camera

RAIZ3_2 = math.sqrt(3) / 2
N30 = (0, 0.5, RAIZ3_2)
JANELA = (-4, -3, 4, 3)
VIEWPORT = (0, 0, 800, 600)


def aprox(v, tol=1e-9):
    return pytest.approx(v, abs=tol)


def cam30(vrp=(0, 0, 0)):
    return camera.CameraOrtografica(vrp, N30, (0, 1, 0), JANELA, VIEWPORT)


def test_base_30_graus():
    u, v, n = camera.base_do_observador(N30, (0, 1, 0))
    assert u == aprox([1, 0, 0])
    assert v == aprox([0, RAIZ3_2, -0.5])
    assert n == aprox([0, 0.5, RAIZ3_2])


def test_base_e_ortonormal_e_mao_direita():
    u, v, n = camera.base_do_observador((1, 2, 3), (0, 1, 0))
    assert np.dot(u, u) == aprox(1) and np.dot(v, v) == aprox(1) and np.dot(n, n) == aprox(1)
    assert np.dot(u, v) == aprox(0) and np.dot(u, n) == aprox(0) and np.dot(v, n) == aprox(0)
    assert np.cross(u, v) == aprox(n)


def test_base_vista_frontal():
    # N = +z e view up = +y: SRC igual ao mundo
    u, v, n = camera.base_do_observador((0, 0, 5), (0, 1, 0))
    assert u == aprox([1, 0, 0]) and v == aprox([0, 1, 0]) and n == aprox([0, 0, 1])


def test_view_up_nao_perpendicular_e_projetado():
    # view up (0,1,1) com N = +z: o componente ao longo de N sai, sobra (0,1,0)
    u, v, n = camera.base_do_observador((0, 0, 1), (0, 1, 1))
    assert v == aprox([0, 1, 0])


def test_view_up_paralelo_a_n_e_erro():
    with pytest.raises(ValueError):
        camera.base_do_observador((0, 1, 0), (0, 2, 0))


def test_janela_viewport_y_invertido_cantos_e_centro():
    m = camera.matriz_janela_viewport(JANELA, VIEWPORT)
    assert camera.mat.aplicar(m, (-4, -3, 0)) == aprox([0, 600, 0])   # canto inferior esquerdo
    assert camera.mat.aplicar(m, (4, 3, 0)) == aprox([800, 0, 0])     # canto superior direito
    assert camera.mat.aplicar(m, (0, 0, 0)) == aprox([400, 300, 0])   # centro
    # (2,1): u=(2+4)*100=600; v=600-(1+3)*100=200
    assert camera.mat.aplicar(m, (2, 1, 0)) == aprox([600, 200, 0])


def test_janela_viewport_y_mesmo_sentido():
    m = camera.matriz_janela_viewport(JANELA, VIEWPORT, y_invertido=False)
    assert camera.mat.aplicar(m, (-4, -3, 0)) == aprox([0, 0, 0])
    assert camera.mat.aplicar(m, (2, 1, 0)) == aprox([600, 400, 0])


def test_janela_viewport_escala_nao_uniforme():
    # janela (0,0)-(10,5) -> viewport (100,50)-(300,250): sx=200/10=20, sy=200/5=40
    m = camera.matriz_janela_viewport((0, 0, 10, 5), (100, 50, 300, 250))
    assert camera.mat.aplicar(m, (10, 0, 0)) == aprox([300, 250, 0])   # v = 250 - 0*40
    assert camera.mat.aplicar(m, (5, 2.5, 0)) == aprox([200, 150, 0])  # v = 250 - 2.5*40


def test_janela_viewport_preserva_profundidade():
    m = camera.matriz_janela_viewport(JANELA, VIEWPORT)
    assert camera.mat.aplicar(m, (1, 1, 7.5))[2] == aprox(7.5)


def test_janela_degenerada_e_erro():
    with pytest.raises(ValueError):
        camera.matriz_janela_viewport((1, 0, 1, 5), VIEWPORT)


def test_ponto_no_eixo_x():
    # (2,0,0): x_c=2, y_c=0, z_c=0  ->  px=(2+4)*100=600, py=600-(0+3)*100=300
    assert cam30().projetar((2, 0, 0)) == aprox([600, 300, 0])


def test_ponto_recuado_em_z():
    # (0,0,1): y_c=v.P=-0.5, z_c=n.P=raiz3/2  ->  py=600-(-0.5+3)*100=350
    assert cam30().projetar((0, 0, 1)) == aprox([400, 350, RAIZ3_2])


def test_ponto_no_eixo_y():
    # (0,1,0): y_c=raiz3/2, z_c=0.5  ->  py=600-(0.8660254+3)*100=213.39746
    assert cam30().projetar((0, 1, 0)) == aprox([400, 600 - (RAIZ3_2 + 3) * 100, 0.5])


def test_vrp_vai_para_o_centro_da_tela():
    c = cam30(vrp=(1, 2, 3))
    assert c.projetar((1, 2, 3)) == aprox([400, 300, 0])


def test_vista_frontal_sem_elevacao():
    # N=+z: SRC = mundo. (2,1,5) -> px=600, py=600-(1+3)*100=200, profundidade=5
    c = camera.CameraOrtografica((0, 0, 0), (0, 0, 1), (0, 1, 0), JANELA, VIEWPORT)
    assert c.projetar((2, 1, 5)) == aprox([600, 200, 5])


def test_projecao_ortografica_andar_ao_longo_de_n_nao_muda_o_pixel():
    c = cam30()
    p = np.array([1.5, -0.4, 2.0])
    antes = c.projetar(p)
    depois = c.projetar(p + 3.0 * c.n)
    assert depois[:2] == aprox(antes[:2])
    assert depois[2] == aprox(antes[2] + 3.0)


def test_eixo_x_do_mundo_fica_horizontal_na_tela():
    c = cam30()
    pts = np.array([[x, 0.7, -1.2] for x in (-3, -1, 0, 2, 5)], float)
    r = c.projetar(pts)
    assert np.allclose(r[:, 1], r[0, 1], atol=1e-9)
    assert np.allclose(np.diff(r[:, 0]), np.diff(pts[:, 0]) * 100, atol=1e-9)  # 100 px por unidade


def test_varios_pontos_de_uma_vez():
    r = cam30().projetar(np.array([[2, 0, 0], [0, 0, 1]], float))
    assert r == aprox(np.array([[600, 300, 0], [400, 350, RAIZ3_2]]))


def test_mais_proximo_do_observador_tem_profundidade_maior():
    c = cam30()
    longe = c.projetar((0, 0, -3))[2]
    perto = c.projetar((0, 0, 3))[2]
    assert perto > longe


# Acrescente 2 a 3 casos seus: (vrp, n, view_up, janela, viewport, ponto, (px, py, profundidade)).
CASOS_DO_USUARIO = []


@pytest.mark.parametrize("vrp, n, up, janela, viewport, ponto, esperado", CASOS_DO_USUARIO)
def test_casos_do_usuario(vrp, n, up, janela, viewport, ponto, esperado):
    c = camera.CameraOrtografica(vrp, n, up, janela, viewport)
    assert c.projetar(ponto) == aprox(esperado)
