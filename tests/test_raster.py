"""Testes do preenchimento por scanline (parte 5 do plano).

Regra (apostila 6.9): o pixel (x, y) e amostrado na coordenada inteira; linha de varredura y vale se
ymin <= y < ymax e, na linha, pinta-se de ceil(xesq) ate ceil(xdir) - 1. Ou seja, os lados esquerdo e de
y minimo pertencem ao poligono; os de x maximo e y maximo, nao. Contagens calculadas a mao.
Conferir as contas e acrescentar casos proprios em CASOS_DO_USUARIO.
"""
import math

import numpy as np
import pytest

from trex3d import raster

LARGURA, ALTURA = 20, 20


def aprox(v, tol=1e-9):
    return pytest.approx(v, abs=tol)


def pixels(P, A=None, w=LARGURA, h=ALTURA, **kw):
    px, py, poly, attrs = raster.rasterizar(np.asarray(P, float)[None] if np.ndim(P) == 2 else P, A, w, h, **kw)
    return px, py, poly, attrs


def conjunto(px, py):
    return set(zip(px.tolist(), py.tolist()))


def test_quadrado_inteiro():
    # (2,3)-(6,7): x em [2,6), y em [3,7) -> 4 x 4 = 16 pixels
    px, py, _, _ = pixels([(2, 3), (6, 3), (6, 7), (2, 7)])
    assert conjunto(px, py) == {(x, y) for x in range(2, 6) for y in range(3, 7)}
    assert px.size == 16


def test_triangulo_retangulo():
    # (0,0),(4,0),(0,4): linha y tem x em [0, 4-y) -> 4+3+2+1 = 10
    px, py, _, _ = pixels([(0, 0), (4, 0), (0, 4)])
    assert px.size == 10
    assert conjunto(px, py) == {(x, y) for y in range(4) for x in range(4 - y)}


def test_triangulo_com_vertices_fracionarios():
    # (0.5,0.5),(4.5,0.5),(0.5,4.5): linhas y = 1..4; esquerda x=0.5 -> 1; direita x=5-y
    # y=1: [1,4) 3 | y=2: [1,3) 2 | y=3: [1,2) 1 | y=4: [1,1) 0  -> 6 pixels
    px, py, _, _ = pixels([(0.5, 0.5), (4.5, 0.5), (0.5, 4.5)])
    assert px.size == 6
    assert conjunto(px, py) == {(1, 1), (2, 1), (3, 1), (1, 2), (2, 2), (1, 3)}


def test_poligono_concavo_em_L():
    # L: (0,0),(4,0),(4,2),(2,2),(2,4),(0,4); y=0,1: [0,4) -> 8 ; y=2,3: [0,2) -> 4 ; total 12 = area
    px, py, _, _ = pixels([(0, 0), (4, 0), (4, 2), (2, 2), (2, 4), (0, 4)])
    assert px.size == 12
    esperado = {(x, y) for y in (0, 1) for x in range(4)} | {(x, y) for y in (2, 3) for x in range(2)}
    assert conjunto(px, py) == esperado


def test_poligono_degenerado_nao_pinta_nada():
    assert pixels([(1, 1), (5, 1), (9, 1)])[0].size == 0       # todos na mesma linha
    assert pixels([(1, 1), (1, 1), (1, 1)])[0].size == 0       # um ponto


def test_triangulo_com_vertice_repetido_igual_ao_triangulo():
    a = pixels([(0, 0), (4, 0), (0, 4)])
    b = pixels([(0, 0), (4, 0), (0, 4), (0, 4)])
    assert conjunto(a[0], a[1]) == conjunto(b[0], b[1])


def test_dois_quadrados_lado_a_lado():
    # (0,0)-(4,4) e (4,0)-(8,4): 8 x 4 = 32 pixels, cada um uma vez; a coluna x = 4 e do segundo
    P = np.array([[(0, 0), (4, 0), (4, 4), (0, 4)], [(4, 0), (8, 0), (8, 4), (4, 4)]], float)
    px, py, poly, _ = raster.rasterizar(P, None, LARGURA, ALTURA)
    assert px.size == 32 and len(conjunto(px, py)) == 32
    assert all(poly[(px == 4)] == 1)


def test_quadrado_dividido_pela_diagonal():
    # triangulo 1 tem 10 pixels (y: 4,3,2,1) e o 2 tem 6 (y: 0,1,2,3): 16 = 4 x 4, sem sobra nem falta
    P = np.array([[(0, 0), (4, 0), (0, 4)], [(4, 0), (4, 4), (0, 4)]], float)
    P = np.concatenate([P, P[:, -1:, :]], axis=1)    # completa os triangulos com o ultimo vertice repetido
    px, py, poly, _ = raster.rasterizar(P, None, LARGURA, ALTURA)
    assert (poly == 0).sum() == 10 and (poly == 1).sum() == 6
    assert conjunto(px, py) == {(x, y) for x in range(4) for y in range(4)}
    assert px.size == 16


def test_recorte_na_borda_esquerda():
    # quadrado x em [-3,3), y em [2,5), tela 10 x 10: so x em [0,3) -> 3 x 3 = 9
    px, py, _, _ = pixels([(-3, 2), (3, 2), (3, 5), (-3, 5)], w=10, h=10)
    assert px.size == 9 and px.min() == 0 and px.max() == 2


def test_recorte_nas_bordas_direita_e_inferior():
    # x em [7,15), y em [8,14), tela 10 x 10: x 7..9 (3) e y 8..9 (2) -> 6
    px, py, _, _ = pixels([(7, 8), (15, 8), (15, 14), (7, 14)], w=10, h=10)
    assert px.size == 6


def test_poligono_totalmente_fora():
    assert pixels([(30, 30), (40, 30), (40, 40), (30, 40)], w=10, h=10)[0].size == 0
    assert pixels([(-9, -9), (-1, -9), (-1, -1), (-9, -1)], w=10, h=10)[0].size == 0


def test_lista_vazia():
    px, py, poly, attrs = raster.rasterizar(np.zeros((0, 4, 2)), None, 10, 10)
    assert px.size == py.size == poly.size == 0


def test_interpolacao_exata_de_funcao_linear_no_triangulo():
    # f(x,y) = 3x - y + 2 nos vertices (0,0),(8,0),(0,8) -> f = 2, 26, -6; em todo pixel f deve valer 3x - y + 2
    P = [(0, 0), (8, 0), (0, 8)]
    A = np.array([[[2.0], [26.0], [-6.0]]])
    px, py, _, a = pixels(np.array([P], float), A)
    assert px.size > 0
    assert a[:, 0] == aprox(3.0 * px - py + 2.0)


def test_interpolacao_no_quadrado_com_dois_atributos():
    # atributos (x, 2y): lados verticais e interpolacao horizontal reproduzem exatamente
    P = np.array([[(0, 0), (6, 0), (6, 5), (0, 5)]], float)
    A = np.stack([P[..., 0], 2 * P[..., 1]], axis=-1)
    px, py, _, a = raster.rasterizar(P, A, LARGURA, ALTURA)
    assert a[:, 0] == aprox(px.astype(float)) and a[:, 1] == aprox(2.0 * py)


def test_atributo_nao_interpolado_e_o_do_inicio_do_trecho():
    P = np.array([[(0, 0), (4, 0), (4, 4), (0, 4)]], float)
    A = np.full((1, 4, 1), 7.0)
    px, py, _, a = raster.rasterizar(P, A, LARGURA, ALTURA, interpolar=False)
    assert np.all(a == 7.0)


def test_preencher_pinta_so_o_poligono():
    fb = np.zeros((10, 10, 3), np.uint8)
    P = np.array([[(2, 3), (6, 3), (6, 7), (2, 7)]], float)
    n = raster.preencher(fb, P, np.array([[10, 20, 30]], np.uint8))
    assert n == 16
    assert np.all(fb[2:6, 3:7] == (10, 20, 30))
    assert fb.reshape(-1, 3).any(axis=1).sum() == 16


def test_preencher_ordem_do_pintor_o_ultimo_cobre():
    fb = np.zeros((10, 10, 3), np.uint8)
    P = np.array([[(0, 0), (6, 0), (6, 6), (0, 6)], [(3, 3), (9, 3), (9, 9), (3, 9)]], float)
    raster.preencher(fb, P, np.array([[255, 0, 0], [0, 255, 0]], np.uint8))
    assert tuple(fb[1, 1]) == (255, 0, 0)
    assert tuple(fb[4, 4]) == (0, 255, 0)       # sobreposicao: vence o segundo
    assert tuple(fb[8, 8]) == (0, 255, 0)
    assert tuple(fb[8, 1]) == (0, 0, 0)


def dentro_por_paridade(poligono, x, y):
    """Pixel (x, y) esta dentro se o raio para a direita cruza um numero impar de arestas
    (regra [ymin, ymax) para as linhas; cruzamento so conta se estritamente a direita de x)."""
    cruzamentos = 0
    n = len(poligono)
    for i in range(n):
        (x1, y1), (x2, y2) = poligono[i], poligono[(i + 1) % n]
        if (y1 <= y < y2) or (y2 <= y < y1):
            xc = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if xc > x:
                cruzamentos += 1
    return cruzamentos % 2 == 1


def poligono_estrelado(rng, n, cx, cy, rmax):
    angulos = np.sort(rng.uniform(0, 2 * math.pi, n))
    raios = rng.uniform(0.3 * rmax, rmax, n)
    return [(cx + r * math.cos(a), cy + r * math.sin(a)) for a, r in zip(angulos, raios)]


@pytest.mark.parametrize("semente", range(12))
def test_poligonos_aleatorios_batem_com_a_regra_de_paridade(semente):
    rng = np.random.default_rng(semente)
    n = int(rng.integers(3, 10))
    pol = poligono_estrelado(rng, n, 20.3, 19.7, 17.9)     # concavo na maioria dos casos
    px, py, _, _ = raster.rasterizar(np.array([pol]), None, 40, 40)
    obtido = conjunto(px, py)
    esperado = {(x, y) for x in range(40) for y in range(40) if dentro_por_paridade(pol, x, y)}
    assert obtido == esperado
    assert px.size == len(obtido)    # nenhum pixel repetido


# Acrescente 2 a 3 casos seus: (lista de vertices, numero de pixels pintados).
CASOS_DO_USUARIO = []


@pytest.mark.parametrize("vertices, n_pixels", CASOS_DO_USUARIO)
def test_casos_do_usuario(vertices, n_pixels):
    assert pixels(vertices)[0].size == n_pixels
