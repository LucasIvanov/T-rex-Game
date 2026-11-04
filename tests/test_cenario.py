"""Testes do cenario e dos extras (parte 10): nuvens fixas e cactos de fundo em escalas diferentes.

Contas a mao:
  caixa (2, 4, 6): meios-lados (1, 2, 3); centro da face +y = (0, 2, 0), da face +z = (0, 0, 3).
  cacto (escala 1): tronco 0.35 x 2.0 x 0.35 centrado em (0, 1, 0); dois bracos horizontais 0.425 x 0.25 x 0.25
    em x = -+0.3875, y = 0.9; dois bracos verticais 0.25 x 0.7 x 0.25 em x = -+0.475, y = 1.125
    -> x em [-0.6, 0.6], y em [0, 2.0], z em [-0.175, 0.175]; 5 caixas = 120 vertices e 30 faces.
  topo do tronco de um cacto na posicao (x, z) com escala s, na tela (camera com 20 graus de elevacao):
    py = 300 - 55 (2 s cos20 + (0.175 s - z) sen20), coluna px = 400 + 55 x.
Conferir as contas e acrescentar casos proprios em CASOS_DO_USUARIO.
"""
import math

import numpy as np
import pytest

from trex3d import cena, cenario, jogo, luz, mat, modelos

ELEV = math.radians(20)


def aprox(v, tol=1e-9):
    return pytest.approx(v, abs=tol)


def eixos(malha):
    return {tuple(int(round(x)) for x in n) for n in malha.normais_face}


def test_caixa_dimensoes_e_faces():
    c = modelos.criar_caixa(2, 4, 6)
    assert c.vertices.shape == (24, 3) and c.faces.shape == (6, 4)
    assert np.abs(c.vertices).max(axis=0) == aprox([1, 2, 3])
    assert eixos(c) == {(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)}


def test_caixa_centros_das_faces():
    c = modelos.criar_caixa(2, 4, 6)
    centros = {tuple(np.round(p, 9)) for p in c.centros_das_faces}
    assert centros == {(1, 0, 0), (-1, 0, 0), (0, 2, 0), (0, -2, 0), (0, 0, 3), (0, 0, -3)}


def test_caixa_normais_para_fora():
    c = modelos.criar_caixa(2, 4, 6)
    assert np.sum(c.normais_face * c.centros_das_faces, axis=1).min() > 0


def test_cubo_e_caixa_com_tres_lados_iguais():
    a, b = modelos.criar_cubo(1.2), modelos.criar_caixa(1.2, 1.2, 1.2)
    assert a.vertices == aprox(b.vertices) and np.array_equal(a.faces, b.faces)


def test_caixa_lado_invalido():
    with pytest.raises(ValueError):
        modelos.criar_caixa(1, 0, 1)


def test_combinar_duas_caixas_deslocadas():
    cubo = modelos.criar_cubo(2)
    m = modelos.combinar([(cubo, mat.translacao(5, 0, 0)), (cubo, mat.translacao(-5, 0, 0))])
    assert m.vertices.shape == (48, 3) and m.faces.shape == (12, 4)
    assert set(np.unique(m.vertices[:24, 0])) == {4.0, 6.0} and set(np.unique(m.vertices[24:, 0])) == {-6.0, -4.0}
    assert m.faces[6:].min() == 24 and m.faces[:6].max() == 23          # indices da 2a parte deslocados
    centros_das_caixas = np.array([(5, 0, 0)] * 6 + [(-5, 0, 0)] * 6, float)
    assert np.sum(m.normais_face * (m.centros_das_faces - centros_das_caixas), axis=1).min() > 0


def test_combinar_gira_as_normais():
    # caixa (2,4,6) girada 90 graus em z: x <- -y: extensoes viram x: +-2, y: +-1; a face +y (0,1,0) vira (-1,0,0)
    c = modelos.criar_caixa(2, 4, 6)
    m = modelos.combinar([(c, mat.rot_z(math.pi / 2))])
    assert np.abs(m.vertices).max(axis=0) == aprox([2, 1, 3])
    i = int(np.argmin(m.centros_das_faces[:, 0]))                      # face em x = -2
    assert m.centros_das_faces[i] == aprox([-2, 0, 0]) and m.normais_face[i] == aprox([-1, 0, 0])
    assert np.linalg.norm(m.normais_vertice, axis=1) == aprox(np.ones(24))


def test_combinar_escala_nao_uniforme_e_erro():
    with pytest.raises(ValueError):
        modelos.combinar([(modelos.criar_cubo(2), mat.escala(2, 1, 1))])


def test_cacto_contagens_e_caixa_envolvente():
    c = modelos.criar_cacto()
    assert c.vertices.shape == (120, 3) and c.faces.shape == (30, 4)
    assert c.vertices.min(axis=0) == aprox([-0.6, 0.0, -0.175])
    assert c.vertices.max(axis=0) == aprox([0.6, 2.0, 0.175])


def test_cacto_faces_alinhadas_aos_eixos():
    assert eixos(modelos.criar_cacto()) <= {(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)}


def test_nuvens_sao_poligonos_acima_do_horizonte():
    P, cores = cenario.poligonos_das_nuvens()
    assert P.ndim == 3 and P.shape[2] == 2 and cores.shape == (P.shape[0], 3)
    assert P[..., 1].max() < 187                      # horizonte do terreno na tela (z = -6: py = 187)
    assert P[..., 0].min() >= 0 and P[..., 0].max() <= cena.LARGURA


def test_nuvens_sao_convexas_e_nao_vazias():
    from trex3d import raster
    P, _ = cenario.poligonos_das_nuvens()
    px, py, poly, _ = raster.rasterizar(P, None, cena.LARGURA, cena.ALTURA)
    assert np.bincount(poly, minlength=P.shape[0]).min() > 100


def test_ha_pelo_menos_3_escalas_diferentes():
    escalas = {e for _, _, e in cenario.CACTOS_DE_FUNDO}
    assert len(escalas) >= 3 and min(escalas) < 1 < max(escalas)


def test_cactos_no_fundo_e_no_terreno():
    for x, z, e in cenario.CACTOS_DE_FUNDO:
        assert -6 < z <= -2.5 and abs(x) <= 7.5        # atras da pista (z = 0) e dentro do terreno


def quadro(j=None):
    j = j or jogo.Jogo()
    c = cena.Cena(j)
    fb = np.zeros((cena.LARGURA, cena.ALTURA, 3), np.uint8)
    c.desenhar(fb, "phong")
    return j, c, fb


def primeira_linha_nao_ceu(fb, coluna, ate=300):
    for y in range(ate):
        if tuple(fb[coluna, y]) != cena.COR_CEU:
            return y
    return None


def topo_esperado(x, z, s):
    return 300 - 55 * (2 * s * math.cos(ELEV) + (0.175 * s - z) * math.sin(ELEV))


def test_cacto_maior_e_mais_alto_na_tela_na_proporcao_da_escala():
    j, c, fb = quadro()
    mais_alto = max(cenario.CACTOS_DE_FUNDO, key=lambda t: t[2])
    mais_baixo = min(cenario.CACTOS_DE_FUNDO, key=lambda t: t[2])
    for x, z, s in (mais_alto, mais_baixo):
        col = round(400 + 55 * x)
        y = primeira_linha_nao_ceu(fb, col)
        assert y is not None and abs(y - topo_esperado(x, z, s)) <= 1.5


def test_nuvens_ficam_fixas_enquanto_o_sol_anda():
    j1, c1, fb1 = quadro()
    j2 = jogo.Jogo()
    j2.mundo.avancar_sol(20.0)
    j2, c2, fb2 = quadro(j2)
    P, _ = cenario.poligonos_das_nuvens()
    for p in P[::3]:                                   # centro de cada nuvem (varios poligonos por nuvem)
        cx, cy = int(p[:, 0].mean()), int(p[:, 1].mean())
        assert tuple(fb1[cx, cy]) == tuple(fb2[cx, cy]) == cenario.COR_NUVEM


def test_cactos_sao_iluminados_pelo_sol():
    # a face da frente (+z) do tronco muda de cor quando o Sol anda
    x, z, s = max(cenario.CACTOS_DE_FUNDO, key=lambda t: t[2])
    col = round(400 + 55 * x)
    y = round(topo_esperado(x, z, s)) + 40
    a = quadro()[2][col, y].astype(int)
    j = jogo.Jogo()
    j.mundo.avancar_sol(50.0)
    b = quadro(j)[2][col, y].astype(int)
    assert tuple(a) != cena.COR_CEU and np.abs(a - b).max() >= 2


def test_material_do_cacto_e_valido():
    m = luz.MATERIAL_CACTO
    for c in (m.ka, m.kd, m.ks):
        assert np.all((c >= 0) & (c <= 1))
    assert m.kd[1] > m.kd[0] and m.kd[1] > m.kd[2]


# Acrescente 2 a 3 casos seus: (lados da caixa (lx, ly, lz), numero de vertices, numero de faces).
CASOS_DO_USUARIO = []


@pytest.mark.parametrize("lados, n_vertices, n_faces", CASOS_DO_USUARIO)
def test_casos_do_usuario(lados, n_vertices, n_faces):
    m = modelos.criar_caixa(*lados)
    assert m.vertices.shape[0] == n_vertices and m.faces.shape[0] == n_faces
