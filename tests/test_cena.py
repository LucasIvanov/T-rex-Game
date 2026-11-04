"""Testes da montagem do quadro (parte 9: a partida aparece na tela).

Cena de 800 x 450 px, 55 px por unidade, camera ortografica com 20 graus de elevacao e o ponto (0,0,0) do
mundo em (400, 300): px = 400 + 55 x; py = 300 - 55 (y cos20 - z sen20).
Conferir as contas e acrescentar casos proprios em CASOS_DO_USUARIO.
"""
import math

import numpy as np
import pytest

from trex3d import cena, jogo, luz, mundo, sombreamento

ELEV = math.radians(20)


def aprox(v, tol=1e-9):
    return pytest.approx(v, abs=tol)


def pixel(x, y, z):
    return round(400 + 55 * x), round(300 - 55 * (y * math.cos(ELEV) - z * math.sin(ELEV)))


def quadro(modo="phong", antes=None):
    j = jogo.Jogo()
    if antes:
        antes(j)
    c = cena.Cena(j)
    fb = np.zeros((cena.LARGURA, cena.ALTURA, 3), np.uint8)
    c.desenhar(fb, modo)
    return j, c, fb


def test_camera_leva_a_origem_ao_centro_inferior_da_tela():
    c = cena.Cena(jogo.Jogo())
    assert c.camera.projetar((0, 0, 0))[:2] == aprox([400, 300])
    assert c.camera.projetar((1, 2, 3))[:2] == aprox(pixel(1, 2, 3), tol=0.5)


@pytest.mark.parametrize("modo", sombreamento.MODOS)
def test_desenha_nos_tres_modos(modo):
    j, c, fb = quadro(modo)
    assert tuple(fb[0, 0]) == cena.COR_CEU
    assert tuple(fb[400, 440]) != cena.COR_CEU
    assert (fb != 0).any(axis=-1).all()


def test_sol_aparece_amarelo_na_posicao_projetada():
    j, c, fb = quadro()
    sx, sy = pixel(*j.mundo.posicao_do_sol())
    assert tuple(fb[sx, sy]) == tuple(luz.para_uint8(luz.MATERIAL_SOL.emissao))


def test_cubo_aparece_na_posicao_esperada():
    j, c, fb = quadro()
    cx, cy = pixel(j.mundo.p.cubo_x, j.mundo.cubo_y, 0)
    assert tuple(fb[cx, cy]) != cena.COR_CEU
    # cubo sobre o chao nao toca o ceu em seu centro e e mais escuro/cinza que o terreno
    assert tuple(fb[cx, cy]) != tuple(fb[400, 440])


def test_sem_obstaculos_no_inicio_e_com_obstaculos_depois():
    def avancar_partida(j):
        j.avancar(0.0, espaco=True)
        for _ in range(130):
            j.avancar(1 / 60)

    j1, c1, fb1 = quadro()
    j2, c2, fb2 = quadro(antes=avancar_partida)
    assert len(j1.mundo.esferas) == 0 and len(j2.mundo.esferas) == 1
    x, y, _ = j2.mundo.estado_da_esfera(j2.mundo.esferas[0])
    px, py = pixel(x, y, 0)
    # onde a esfera esta no segundo quadro, o primeiro mostra o terreno; o segundo mostra a esfera (verde)
    assert tuple(fb1[px, py]) == tuple(fb1[400, 440])
    f = fb2[px, py].astype(int)
    assert f[1] > f[0] and f[1] > f[2]


def test_o_sol_andando_muda_a_cor_do_terreno():
    def sol_no_meio(j):
        j.mundo.avancar_sol(25.0)

    _, _, antes = quadro("constante")
    _, _, depois = quadro("constante", antes=sol_no_meio)
    cor = lambda fb: fb[400, 440].astype(int)
    assert np.abs(cor(antes) - cor(depois)).max() >= 3


def test_terreno_fica_atras_dos_objetos():
    # uma esfera sobre o terreno: o pixel do centro da esfera nao e a cor do terreno
    def com_esfera(j):
        j.avancar(0.0, espaco=True)
        j.mundo.esferas.append(mundo.Esfera(t_spawn=j.mundo.t_jogo - 0.5, fase=0.0))

    j, c, fb = quadro(antes=com_esfera)
    x, y, _ = j.mundo.estado_da_esfera(j.mundo.esferas[0])
    assert tuple(fb[pixel(x, y, 0)]) != tuple(fb[400, 440])


# Acrescente 2 a 3 casos seus: (ponto do mundo (x, y, z), pixel (px, py) esperado), na camera acima.
CASOS_DO_USUARIO = []


@pytest.mark.parametrize("ponto, esperado", CASOS_DO_USUARIO)
def test_casos_do_usuario(ponto, esperado):
    assert cena.Cena(jogo.Jogo()).camera.projetar(ponto)[:2] == aprox(esperado, tol=0.5)
