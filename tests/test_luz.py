"""Testes da iluminacao (parte 6 do plano).

Modelo (apostila 8.1): I = ka*Ia + fatt*Il*(kd*max(0, N.L) + ks*max(0, R.S)^n) + emissao, por canal RGB,
com L = (Sol - P)/|Sol - P|, R = 2 (N.L) N - L, fatt = min(1/(c1 + c2 d + c3 d^2), 1). O brilho
especular so existe se N.L > 0. Resultado cortado em [0, 1]. Contas feitas a mao nos comentarios.

Material de referencia M: ka = 0.2 (cinza), kd = (0.5, 0.4, 0.3), ks = 0.3 (cinza), n = 2.
Conferir as contas e acrescentar casos proprios em CASOS_DO_USUARIO.
"""
import math

import numpy as np
import pytest

from trex3d import luz

M = luz.Material(ka=(0.2, 0.2, 0.2), kd=(0.5, 0.4, 0.3), ks=(0.3, 0.3, 0.3), n=2)
N_Z = (0, 0, 1)     # normal da face
S_Z = (0, 0, 1)     # direcao do observador (ortografica, constante)
ORIGEM = (0, 0, 0)


def aprox(v, tol=1e-9):
    return pytest.approx(v, abs=tol)


def sol(x, y, z, **kw):
    return luz.Luz(posicao=(x, y, z), **kw)


def ilum(normal, luz_, material=M, ponto=ORIGEM, s=S_Z):
    return luz.iluminar(ponto, normal, material, luz_, s)


def test_face_de_frente_para_o_sol_e_o_observador():
    # L = (0,0,1): N.L = 1; R = (0,0,1): R.S = 1 -> I = ka + kd + ks = (1.0, 0.9, 0.8)
    assert ilum(N_Z, sol(0, 0, 10)) == aprox([1.0, 0.9, 0.8])


def test_face_de_costas_para_o_sol_so_tem_ambiente():
    # N = (0,0,-1): N.L = -1 -> difusa 0; sem o corte, R.S seria 1 e somaria ks; esperado so ka
    assert ilum((0, 0, -1), sol(0, 0, 10)) == aprox([0.2, 0.2, 0.2])


def test_luz_rasante_so_tem_ambiente():
    # Sol em (10,0,0): N.L = 0
    assert ilum(N_Z, sol(10, 0, 0)) == aprox([0.2, 0.2, 0.2])


def test_incidencia_de_60_graus():
    # L = (sen60, 0, 0.5): N.L = 0.5; R = (-sen60, 0, 0.5): R.S = 0.5; especular = 0.5^2 = 0.25
    # I = 0.2 + kd*0.5 + 0.3*0.25 = (0.2+0.25+0.075, 0.2+0.2+0.075, 0.2+0.15+0.075)
    s60 = math.sin(math.radians(60))
    assert ilum(N_Z, sol(10 * s60, 0, 5)) == aprox([0.525, 0.475, 0.425])


def test_observador_na_direcao_do_reflexo_pega_o_brilho_maximo():
    # L = (0.6,0,0.8): N.L = 0.8; R = (-0.6,0,0.8); com S = R: R.S = 1
    # I = 0.2 + kd*0.8 + 0.3*1 = (0.9, 0.82, 0.74)
    assert ilum(N_Z, sol(6, 0, 8), s=(-0.6, 0, 0.8)) == aprox([0.9, 0.82, 0.74])


def test_observador_fora_do_reflexo_so_tem_a_difusa():
    # mesmo Sol, S = (0.6,0,0.8): R.S = -0.36 + 0.64 = 0.28; especular = 0.28^2 = 0.0784
    # I = 0.2 + kd*0.8 + 0.3*0.0784
    esperado = [0.2 + 0.5 * 0.8 + 0.3 * 0.0784, 0.2 + 0.4 * 0.8 + 0.3 * 0.0784, 0.2 + 0.3 * 0.8 + 0.3 * 0.0784]
    assert ilum(N_Z, sol(6, 0, 8), s=(0.6, 0, 0.8)) == aprox(esperado)


def test_sem_atenuacao_por_padrao_a_distancia_nao_importa():
    assert ilum(N_Z, sol(0, 0, 50)) == aprox(ilum(N_Z, sol(0, 0, 10)))


def test_atenuacao_quadratica():
    # c = (0, 0, 0.01): d = 20 -> fatt = 1/(0.01*400) = 0.25; I = ka + 0.25*(kd + ks) = (0.4, 0.375, 0.35)
    assert ilum(N_Z, sol(0, 0, 20, c1=0, c2=0, c3=0.01)) == aprox([0.4, 0.375, 0.35])


def test_atenuacao_limitada_a_1_perto_da_fonte():
    # d = 5: 1/(0.01*25) = 4 -> min(4, 1) = 1 -> igual ao caso sem atenuacao
    assert ilum(N_Z, sol(0, 0, 5, c1=0, c2=0, c3=0.01)) == aprox([1.0, 0.9, 0.8])


def test_o_ambiente_nao_e_atenuado():
    assert ilum((0, 0, -1), sol(0, 0, 20, c1=0, c2=0, c3=0.01)) == aprox([0.2, 0.2, 0.2])


def test_luz_colorida_age_por_canal():
    # Il = (1, 0.5, 0): R: 0.2 + 0.5 + 0.3 = 1.0; G: 0.2 + (0.4 + 0.3)*0.5 = 0.55; B: 0.2
    assert ilum(N_Z, sol(0, 0, 10, intensidade=(1, 0.5, 0))) == aprox([1.0, 0.55, 0.2])


def test_ambiente_colorido():
    # Ia = (0.5, 1, 0): ka*Ia = (0.1, 0.2, 0); luz de costas -> so isso
    assert ilum((0, 0, -1), sol(0, 0, 10, ambiente=(0.5, 1, 0))) == aprox([0.1, 0.2, 0.0])


def test_resultado_e_cortado_em_1():
    forte = luz.Material(ka=(0.5, 0.5, 0.5), kd=(0.9, 0.9, 0.9), ks=(0.5, 0.5, 0.5), n=1)   # soma 1.9
    assert ilum(N_Z, sol(0, 0, 10), material=forte) == aprox([1.0, 1.0, 1.0])


def test_emissao_independe_da_luz():
    sol_mat = luz.Material(ka=(0, 0, 0), kd=(0, 0, 0), ks=(0, 0, 0), n=1, emissao=(1.0, 0.9, 0.0))
    assert ilum(N_Z, sol(0, 0, 10), material=sol_mat) == aprox([1.0, 0.9, 0.0])
    assert ilum((0, 0, -1), sol(0, 0, 10), material=sol_mat) == aprox([1.0, 0.9, 0.0])


def test_varios_pontos_de_uma_vez():
    pontos = np.array([ORIGEM, ORIGEM], float)
    normais = np.array([N_Z, (0, 0, -1)], float)
    r = luz.iluminar(pontos, normais, M, sol(0, 0, 10), S_Z)
    assert r == aprox(np.array([[1.0, 0.9, 0.8], [0.2, 0.2, 0.2]]))


def test_o_sol_andando_muda_a_cor_da_face():
    # topo (0,1,0) com o Sol subindo da direita; observador em +z, entao R.S = 0 (R fica no plano xy)
    # e so ha ambiente (0.2) + difusa (kd_R = 0.5 * N.L).
    c30, s30 = math.cos(math.radians(30)), math.sin(math.radians(30))
    posicoes = [(10, 0, 0), (10 * c30, 10 * s30, 0), (0, 10, 0)]      # N.L = 0, sen30 = 0.5, 1
    r = [ilum((0, 1, 0), sol(*p), s=(0, 0, 1))[0] for p in posicoes]
    assert r[0] == aprox(0.2)                  # horizonte
    assert r[1] == aprox(0.2 + 0.5 * 0.5)      # 0.45
    assert r[2] == aprox(0.2 + 0.5 * 1.0)      # 0.7


def test_ponto_na_posicao_do_sol_e_erro():
    with pytest.raises(ValueError):
        ilum(N_Z, sol(0, 0, 0))


def test_para_uint8_arredonda_para_o_mais_proximo():
    # 0.2 -> 51; 0.525 -> 133.875 -> 134; 0.475 -> 121.125 -> 121; 0.425 -> 108.375 -> 108; 1.0 -> 255
    r = luz.para_uint8(np.array([[0.2, 0.525, 1.0], [0.475, 0.425, 0.0]]))
    assert r.dtype == np.uint8
    assert r.tolist() == [[51, 134, 255], [121, 108, 0]]


def test_para_uint8_corta_fora_do_intervalo():
    assert luz.para_uint8(np.array([-0.3, 1.7])).tolist() == [0, 255]


def test_material_invalido():
    with pytest.raises(ValueError):
        luz.Material(ka=(0.2, 0.2, 0.2), kd=(0.5, 0.4, 0.3), ks=(0.3, 0.3, 0.3), n=0)
    with pytest.raises(ValueError):
        luz.Material(ka=(-0.1, 0.2, 0.2), kd=(0.5, 0.4, 0.3), ks=(0.3, 0.3, 0.3), n=2)


def test_luz_com_atenuacao_toda_zero_e_erro():
    with pytest.raises(ValueError):
        luz.Luz(posicao=(0, 0, 1), c1=0, c2=0, c3=0)


def test_materiais_do_jogo_tem_coeficientes_validos():
    for m in (luz.MATERIAL_CUBO, luz.MATERIAL_ESFERA, luz.MATERIAL_TERRENO, luz.MATERIAL_SOL):
        for c in (m.ka, m.kd, m.ks, m.emissao):
            assert np.all((c >= 0) & (c <= 1))


def test_sol_e_amarelo_emissivo_e_ignora_a_luz():
    a = ilum(N_Z, sol(0, 0, 10), material=luz.MATERIAL_SOL)
    b = ilum((0, 0, -1), sol(0, 0, 10), material=luz.MATERIAL_SOL)
    assert a == aprox(b)
    assert a[0] > a[1] > a[2] > 0


def test_terreno_e_fosco_e_esfera_e_brilhante():
    assert np.all(luz.MATERIAL_TERRENO.ks == 0)
    assert np.all(luz.MATERIAL_ESFERA.ks > 0)


def test_esfera_fica_mais_clara_de_frente_para_o_sol_do_que_de_lado():
    de_frente = ilum(N_Z, sol(0, 0, 10), material=luz.MATERIAL_ESFERA)
    de_lado = ilum(N_Z, sol(10, 0, 0), material=luz.MATERIAL_ESFERA)
    assert np.all(de_frente > de_lado)


# Acrescente 2 a 3 casos seus: (normal, posicao do Sol, direcao do observador, (R, G, B) esperado) para o
# material de referencia M acima (ka=0.2, kd=(0.5,0.4,0.3), ks=0.3, n=2; ponto na origem; sem atenuacao).
CASOS_DO_USUARIO = []


@pytest.mark.parametrize("normal, posicao_sol, s, esperado", CASOS_DO_USUARIO)
def test_casos_do_usuario(normal, posicao_sol, s, esperado):
    assert ilum(normal, sol(*posicao_sol), s=s) == aprox(esperado)
