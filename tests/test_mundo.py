"""Testes do mundo e da animacao (parte 8 do plano).

Tudo e funcao do tempo (nao do numero de quadros). Parametros padrao e contas a mao:
  salto: altura H = 2.0 em T = 0.8 s -> v0 = 4H/T = 10, g = 8H/T^2 = 25; y(ts) = 10 ts - 12.5 ts^2
         ts = 0.2 -> 1.5 | 0.4 -> 2.0 (topo) | 0.6 -> 1.5 | 0.8 -> 0; o centro do cubo (lado 1.2) fica em 0.6 + y
  esferas: nascem em x = 8 no instante t_spawn; x = 8 - 6 (t - t_spawn); angulo = 6 (t - t_spawn) / 0.6 rad;
           altura do centro = 0.6 + 0.3 |sen(pi (t - t_spawn)/0.8 + fase)|
  Sol: x = 6 - 0.2 t (volta de 8 para -8 ao passar de -8); y = 6.5 + 1.0 cos(pi x / 16); z = 8
  intervalo entre esferas: de T + 0.35 = 1.15 s a 2.15 s; primeira esfera em t = 1.0 s do jogo.
Conferir as contas e acrescentar casos proprios em CASOS_DO_USUARIO.
"""
import math

import numpy as np
import pytest

from trex3d import mat, mundo

BASE = 0.6   # centro do cubo no chao


def aprox(v, tol=1e-9):
    return pytest.approx(v, abs=tol)


def simular(dt, duracao, t_pulo=None, **kw):
    m = mundo.Mundo(mundo.Parametros(**kw))
    quadros = round(duracao / dt)
    k_pulo = None if t_pulo is None else round(t_pulo / dt)
    for k in range(quadros):
        m.avancar(dt, pular=(k == k_pulo))
    return m


def test_parametros_derivados_do_salto():
    p = mundo.Parametros()
    assert p.v0 == aprox(10.0) and p.g == aprox(25.0)
    assert p.intervalo_min == aprox(1.15) and p.intervalo_max == aprox(2.15)


@pytest.mark.parametrize("campo, valor", [("salto_altura", 0), ("salto_duracao", -1), ("velocidade", 0),
                                          ("esfera_raio", 0), ("sol_velocidade", -1), ("quicada_periodo", 0)])
def test_parametros_invalidos(campo, valor):
    with pytest.raises(ValueError):
        mundo.Parametros(**{campo: valor})


def test_sol_e_bem_mais_lento_que_os_obstaculos():
    p = mundo.Parametros()
    assert p.sol_velocidade * 10 <= p.velocidade


def test_cubo_parado_no_chao():
    m = mundo.Mundo()
    m.avancar_jogo(0.5, pular=False)
    assert m.cubo_y == aprox(BASE) and not m.no_ar


def test_salto_subida_topo_descida_e_pouso():
    m = mundo.Mundo()
    m.avancar_jogo(0.2, pular=True)
    assert m.cubo_y == aprox(BASE + 1.5) and m.no_ar
    m.avancar_jogo(0.2)
    assert m.cubo_y == aprox(BASE + 2.0)
    m.avancar_jogo(0.2)
    assert m.cubo_y == aprox(BASE + 1.5)
    m.avancar_jogo(0.2)
    assert m.cubo_y == aprox(BASE) and not m.no_ar       # volta exatamente ao chao


def test_nao_ha_salto_duplo():
    m = mundo.Mundo()
    m.avancar_jogo(0.2, pular=True)
    m.avancar_jogo(0.2, pular=True)                      # apertar de novo no ar nao reinicia
    assert m.cubo_y == aprox(BASE + 2.0)


def test_novo_salto_depois_do_pouso():
    m = mundo.Mundo()
    for pular in (True, False, False, False):
        m.avancar_jogo(0.2, pular=pular)
    assert not m.no_ar
    m.avancar_jogo(0.2, pular=True)
    assert m.no_ar and m.cubo_y == aprox(BASE + 1.5)


def test_pouso_com_passo_grande_nao_afunda_no_chao():
    m = mundo.Mundo()
    m.avancar_jogo(1.0, pular=True)                      # passo maior que a duracao do salto
    assert m.cubo_y == aprox(BASE) and not m.no_ar


def test_matriz_do_cubo():
    m = mundo.Mundo()
    m.avancar_jogo(0.2, pular=True)
    assert mat.aplicar(m.matriz_cubo(), (0, 0, 0)) == aprox([-5.5, BASE + 1.5, 0])


def test_primeira_esfera_nasce_em_1_segundo_na_direita():
    m = mundo.Mundo()
    for _ in range(3):
        m.avancar_jogo(0.25)
    assert m.esferas == []                               # t = 0.75
    m.avancar_jogo(0.25)                                 # t = 1.00: nasce em x = 8
    x, y, theta = m.estado_da_esfera(m.esferas[0])
    assert x == aprox(8.0) and theta == aprox(0.0)


def test_esfera_anda_para_a_esquerda_e_rola():
    m = mundo.Mundo()
    for _ in range(6):
        m.avancar_jogo(0.25)                            # t = 1.5: 0.5 s depois de nascer
    e = m.esferas[0]
    x, y, theta = m.estado_da_esfera(e)
    assert x == aprox(5.0)                               # 8 - 6 * 0.5
    assert theta == aprox(5.0)                           # 6 * 0.5 / 0.6
    assert y == aprox(0.6 + 0.3 * abs(math.sin(math.pi * 0.5 / 0.8 + e.fase)))


def test_quicada_toca_o_chao_e_sobe_no_maximo_0_3():
    p = mundo.Parametros()
    m = mundo.Mundo(p)
    m.esferas.append(mundo.Esfera(t_spawn=0.0, fase=0.0))
    for t_local, esperado in ((0.0, 0.6), (0.4, 0.9), (0.8, 0.6), (1.2, 0.9)):
        m.t_jogo = t_local
        assert m.estado_da_esfera(m.esferas[0])[1] == aprox(esperado)


def test_sem_quicadas_e_sem_rolagem():
    m = mundo.Mundo(mundo.Parametros(quicadas=False, rolagem=False))
    m.esferas.append(mundo.Esfera(t_spawn=0.0, fase=1.0))
    m.t_jogo = 0.7
    x, y, theta = m.estado_da_esfera(m.esferas[0])
    assert y == aprox(0.6) and theta == aprox(0.0) and x == aprox(8 - 6 * 0.7)


def test_matriz_da_esfera_leva_o_centro_e_gira():
    m = mundo.Mundo()
    for _ in range(6):
        m.avancar_jogo(0.25)
    x, y, theta = m.estado_da_esfera(m.esferas[0])
    mt = m.matrizes_esferas()[0]
    assert mat.aplicar(mt, (0, 0, 0)) == aprox([x, y, 0])
    # ponto (r, 0, 0) da esfera, girado de theta, em torno do centro
    r = 0.6
    assert mat.aplicar(mt, (r, 0, 0)) == aprox([x + r * math.cos(theta), y + r * math.sin(theta), 0])


def test_intervalo_entre_esferas_respeita_o_minimo_e_o_maximo():
    m = simular(1 / 60, 120.0)
    t = np.array(m.historico_spawns)
    assert t[0] == aprox(1.0)
    assert len(t) > 40
    d = np.diff(t)
    assert d.min() >= 1.15 - 1e-9 and d.max() <= 2.15 + 1e-9


def test_esferas_fora_da_tela_sao_removidas():
    m = simular(1 / 60, 60.0)
    assert 0 < len(m.esferas) <= 8
    assert all(m.estado_da_esfera(e)[0] >= -8.5 for e in m.esferas)


def test_mesma_semente_mesma_sequencia():
    a = simular(1 / 60, 30.0, semente=7)
    b = simular(1 / 60, 30.0, semente=7)
    c = simular(1 / 60, 30.0, semente=8)
    assert a.historico_spawns == b.historico_spawns
    assert a.historico_spawns != c.historico_spawns


def test_sol_posicao_inicial():
    # x = 6: y = 6.5 + cos(6 pi / 16) = 6.5 + cos(67.5 graus) = 6.8827
    s = mundo.Mundo().posicao_do_sol()
    assert s == aprox([6.0, 6.5 + math.cos(math.radians(67.5)), 8.0])


def test_sol_anda_para_a_esquerda_devagar():
    m = mundo.Mundo()
    m.avancar_sol(10.0)                                  # x = 6 - 0.2 * 10 = 4
    assert m.posicao_do_sol() == aprox([4.0, 6.5 + math.cos(math.radians(45)), 8.0])
    m.avancar_sol(20.0)                                  # t = 30: x = 0, topo do arco
    assert m.posicao_do_sol() == aprox([0.0, 7.5, 8.0])


def test_sol_volta_pela_direita():
    m = mundo.Mundo()
    m.avancar_sol(75.0)                                  # x = 6 - 15 = -9 -> +7
    assert m.posicao_do_sol()[0] == aprox(7.0)


def test_sol_independe_do_jogo():
    parado = mundo.Mundo()
    jogando = mundo.Mundo()
    for k in range(300):
        parado.avancar_sol(1 / 60)
        jogando.avancar(1 / 60, pular=(k % 60 == 0))
    assert jogando.posicao_do_sol() == aprox(parado.posicao_do_sol())


def test_avancar_jogo_nao_mexe_no_sol_e_avancar_sol_nao_mexe_no_jogo():
    m = mundo.Mundo()
    antes = m.posicao_do_sol().copy()
    m.avancar_jogo(2.0, pular=True)
    assert m.posicao_do_sol() == aprox(antes)
    t = m.t_jogo
    m.avancar_sol(5.0)
    assert m.t_jogo == t


def estado(m):
    esf = {round(e.t_spawn, 6): m.estado_da_esfera(e) for e in m.esferas if m.estado_da_esfera(e)[0] > -7}
    return m.cubo_y, esf, m.posicao_do_sol()


def test_mesmo_resultado_a_30_e_a_120_quadros_por_segundo():
    a = estado(simular(1 / 30, 3.0, t_pulo=0.5))
    b = estado(simular(1 / 120, 3.0, t_pulo=0.5))
    assert a[0] == aprox(b[0])
    assert a[2] == aprox(b[2])
    assert set(a[1]) == set(b[1]) and len(a[1]) >= 1
    for chave in a[1]:
        assert a[1][chave] == aprox(b[1][chave])


def test_mesmo_resultado_com_passo_irregular():
    # passos 0.5, 0.1, 0.3, 0.6, 0.5 (t = 0.5, 0.6, 0.9, 1.5, 2.0); salto apertado em t = 0.5
    irregular = mundo.Mundo()
    t = 0.0
    parciais = {}
    for dt in (0.5, 0.1, 0.3, 0.6, 0.5):
        irregular.avancar(dt, pular=(t == 0.5))
        t += dt
        parciais[round(t, 6)] = estado(irregular)
    for instante in (0.9, 2.0):
        regular = estado(simular(1 / 120, instante, t_pulo=0.5))
        obtido = parciais[instante]
        assert obtido[0] == aprox(regular[0])                 # cubo (em 0.9: no topo, BASE + 2.0)
        assert obtido[2] == aprox(regular[2])
        assert set(obtido[1]) == set(regular[1])
        for chave in obtido[1]:
            assert obtido[1][chave] == aprox(regular[1][chave])
    assert parciais[0.9][0] == aprox(BASE + 2.0)
    assert len(parciais[2.0][1]) == 1                         # a esfera que nasceu em t = 1.0


def test_dt_negativo_e_erro_e_dt_zero_nao_muda_nada():
    m = mundo.Mundo()
    with pytest.raises(ValueError):
        m.avancar(-0.1)
    m.avancar_jogo(0.3, pular=True)
    y = m.cubo_y
    m.avancar(0.0)
    assert m.cubo_y == y


def test_reiniciar_jogo_limpa_o_jogo_e_mantem_o_sol():
    m = mundo.Mundo()
    for k in range(240):
        m.avancar(1 / 60, pular=(k == 100))
    sol = m.posicao_do_sol().copy()
    assert m.esferas and m.historico_spawns
    m.reiniciar_jogo()
    assert m.esferas == [] and m.historico_spawns == [] and m.t_jogo == 0.0
    assert not m.no_ar and m.cubo_y == aprox(BASE)
    assert m.posicao_do_sol() == aprox(sol)
    for _ in range(4):
        m.avancar_jogo(0.25)
    assert m.historico_spawns[0] == aprox(1.0)           # o atraso inicial vale de novo


# Acrescente 2 a 3 casos seus: (tempo desde o inicio do salto, altura do cubo acima do chao).
# Com os parametros padrao: y(ts) = 10 ts - 12.5 ts^2 para 0 <= ts <= 0.8.
CASOS_DO_USUARIO = []


@pytest.mark.parametrize("ts, altura", CASOS_DO_USUARIO)
def test_casos_do_usuario(ts, altura):
    m = mundo.Mundo()
    m.avancar_jogo(ts, pular=True)
    assert m.cubo_y == aprox(BASE + altura)
