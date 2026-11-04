"""Testes da mecanica do jogo (parte 9 do plano).

Colisao cubo x esfera: ponto do cubo mais proximo do centro da esfera; ha colisao se a distancia for
MENOR que o raio (encostar nao conta). Cubo de lado 1.2 (meio-lado 0.6) e esfera de raio 0.6:
  - centro da esfera a 1.2 de distancia horizontal: dx = 1.2 - 0.6 = 0.6 = r -> encosta, sem colisao; a 1.1 colide;
  - quina: esfera em (1.0, 1.6) em relacao ao centro do cubo (0, 0.6) -> dx = 0.4, dy = 0.4, dist = 0.566 < 0.6 colide;
    em (1.1, 1.7): dx = dy = 0.5, dist = 0.707 > 0.6 nao colide, embora cada eixo isolado esteja dentro de r.
Primeira esfera: nasce em t = 1.0 s em x = 8; encosta no cubo em x = -5.5 + 0.6 + 0.6 = -4.3, isto e,
t = 1.0 + (8 + 4.3) / 6 = 3.05 s.
Conferir as contas e acrescentar casos proprios em CASOS_DO_USUARIO.
"""
import math

import numpy as np
import pytest

from trex3d import jogo, mundo

P = jogo.PARAMETROS_DO_JOGO
SEM_SPAWN = mundo.Parametros(**{**P.__dict__, "atraso_inicial": 1000.0})
CUBO = (0.0, 0.6, 0.0)


def aprox(v, tol=1e-9):
    return pytest.approx(v, abs=tol)


def colide(sx, sy, sz=0.0, cubo=CUBO):
    return jogo.colisao_cubo_esfera(cubo, 1.2, (sx, sy, sz), 0.6)


def novo_jogo(parametros=None, **kw):
    j = jogo.Jogo(parametros or SEM_SPAWN, **kw)
    j.avancar(0.0, espaco=True)             # inicia a partida (sem pular)
    return j


def com_esfera(j, x, fase=0.0):
    """Coloca uma esfera em x (agora), usando o relogio do jogo."""
    tl = (j.mundo.p.spawn_x - x) / j.mundo.p.velocidade
    j.mundo.esferas.append(mundo.Esfera(t_spawn=j.mundo.t_jogo - tl, fase=fase))


def test_parametros_do_jogo_mais_generosos_que_os_de_referencia():
    # H = 2.4 em T = 0.9: v0 = 4*2.4/0.9, g = 8*2.4/0.9^2; intervalo minimo = 0.9 + 0.35
    assert P.v0 == aprox(4 * 2.4 / 0.9) and P.g == aprox(8 * 2.4 / 0.81)
    assert P.intervalo_min == aprox(1.25)


def test_colisao_horizontal():
    assert colide(1.1, 0.6)
    assert not colide(1.2, 0.6)           # encosta: distancia = r
    assert not colide(2.0, 0.6)


def test_colisao_vertical():
    assert colide(0.0, 1.7)
    assert not colide(0.0, 1.8)           # base da esfera encosta no topo do cubo (1.2)
    assert not colide(0.0, 3.0)


def test_colisao_na_quina_usa_a_distancia_e_nao_cada_eixo():
    assert colide(1.0, 1.6)                # dx = dy = 0.4: dist 0.566 < 0.6
    assert not colide(1.1, 1.7)            # dx = dy = 0.5: dist 0.707 > 0.6


def test_colisao_com_a_esfera_dentro_do_cubo():
    assert colide(0.0, 0.6)


def test_colisao_em_z():
    # cubo ocupa z em [-0.6, 0.6]; esfera em z = 1.0 -> dz = 0.4 colide; em z = 1.3 -> dz = 0.7 nao
    assert colide(0.0, 0.6, 1.0)
    assert not colide(0.0, 0.6, 1.3)


def test_colisao_acompanha_o_deslocamento_do_cubo():
    assert colide(5.1, 0.6, cubo=(4.0, 0.6, 0.0))
    assert not colide(5.1, 0.6)


def test_comeca_esperando_com_o_sol_andando_e_sem_obstaculos():
    j = jogo.Jogo(SEM_SPAWN)
    assert j.estado == "esperando" and j.pontos == 0
    x0 = j.mundo.posicao_do_sol()[0]
    j.avancar(5.0)
    assert j.estado == "esperando"
    assert j.mundo.posicao_do_sol()[0] == aprox(x0 - 0.2 * 5.0)
    assert j.mundo.esferas == [] and j.mundo.t_jogo == 0.0


def test_espaco_inicia_a_partida_sem_pular():
    j = jogo.Jogo(SEM_SPAWN)
    j.avancar(0.1, espaco=True)
    assert j.estado == "jogando" and not j.mundo.no_ar


def test_espaco_pula_durante_a_partida():
    j = novo_jogo()
    j.avancar(0.1, espaco=True)
    assert j.mundo.no_ar and j.mundo.cubo_y > 0.6


def test_primeira_esfera_colide_em_3_05_segundos_se_nao_pular():
    j = jogo.Jogo(P)
    j.avancar(0.0, espaco=True)
    t = 0.0
    while j.estado == "jogando" and t < 10:
        j.avancar(1 / 60)
        t += 1 / 60
    assert j.estado == "fim"
    assert 3.05 - 1e-6 < j.mundo.t_jogo <= 3.05 + 1 / 120 + 1e-6
    assert j.esfera_colidida is not None


def test_passo_enorme_nao_atravessa_o_cubo():
    j = novo_jogo()
    com_esfera(j, -2.0)                  # a 3.5 do cubo; 1 s depois estaria em x = -8, do outro lado
    j.avancar(1.0)
    assert j.estado == "fim"


def test_fim_congela_o_jogo_mas_o_sol_continua():
    j = novo_jogo()
    com_esfera(j, -4.5)
    j.avancar(0.05)
    assert j.estado == "fim"
    t_jogo, sol = j.mundo.t_jogo, j.mundo.posicao_do_sol().copy()
    posicoes = [j.mundo.estado_da_esfera(e) for e in j.mundo.esferas]
    j.avancar(2.0)
    assert j.mundo.t_jogo == t_jogo
    assert [j.mundo.estado_da_esfera(e) for e in j.mundo.esferas] == posicoes
    assert j.mundo.posicao_do_sol()[0] == aprox(sol[0] - 0.2 * 2.0)


def test_reinicio_so_depois_do_atraso_e_zera_tudo():
    j = novo_jogo(atraso_reinicio=0.5)
    j.avancar(0.05, espaco=True)         # comeca um salto curto: o cubo ainda cruza a altura da esfera
    com_esfera(j, -4.5)
    j.avancar(0.05)
    assert j.estado == "fim"
    j.avancar(0.1, espaco=True)
    assert j.estado == "fim"             # ainda dentro do atraso: o espaco e ignorado
    j.avancar(0.5)
    j.avancar(0.0, espaco=True)
    assert j.estado == "jogando"
    assert j.pontos == 0 and j.mundo.t_jogo == 0.0 and j.mundo.esferas == []
    assert not j.mundo.no_ar and j.mundo.cubo_y == aprox(0.6) and j.esfera_colidida is None


def test_ponto_so_quando_a_esfera_passa_inteira_pelo_cubo():
    # passou = centro em x < -5.5 - 0.6 - 0.6 = -6.7; x = -7 passou, x = 2 nao
    j = novo_jogo()
    com_esfera(j, -7.0)
    com_esfera(j, 2.0)
    j.avancar(0.01)
    assert j.pontos == 1
    j.avancar(0.01)
    assert j.pontos == 1                 # a mesma esfera nao conta duas vezes


def test_tempo_de_sobrevivencia():
    j = novo_jogo()
    j.avancar(2.5)
    assert j.tempo == aprox(2.5)


def test_recorde_guarda_a_melhor_pontuacao():
    j = novo_jogo(atraso_reinicio=0.0)
    com_esfera(j, -7.0)
    com_esfera(j, -4.5)
    j.avancar(0.05)
    assert j.estado == "fim" and j.pontos == 1 and j.recorde == 1
    j.avancar(0.0, espaco=True)
    com_esfera(j, -4.5)
    j.avancar(0.05)
    assert j.estado == "fim" and j.pontos == 0 and j.recorde == 1


def test_um_piloto_automatico_sobrevive_40_segundos():
    j = jogo.Jogo(P)
    j.avancar(0.0, espaco=True)
    for _ in range(40 * 60):
        j.avancar(1 / 60, espaco=jogo.piloto_automatico(j))
    assert j.estado == "jogando"
    assert j.pontos >= 15


def janela_de_acerto(fase, passo=0.02):
    """Quantos segundos de variacao no instante do salto ainda passam por cima da esfera."""
    t_centro = (SEM_SPAWN.spawn_x - SEM_SPAWN.cubo_x) / SEM_SPAWN.velocidade     # esfera alinhada com o cubo
    t_ini = t_centro - 1.2
    passagens = int(round(2.2 / passo))
    validos = 0
    for i0 in range(int(0.15 / passo), int(1.1 / passo)):
        j = jogo.Jogo(SEM_SPAWN, atraso_reinicio=1e9)     # nunca reinicia: so conta o primeiro encontro
        j.avancar(0.0, espaco=True)
        j.mundo.t_jogo = t_ini
        j.mundo.esferas.append(mundo.Esfera(t_spawn=0.0, fase=fase))
        for i in range(passagens):
            j.avancar(passo, espaco=(i == i0))
        validos += j.estado == "jogando"
    sem_salto = jogo.Jogo(SEM_SPAWN, atraso_reinicio=1e9)
    sem_salto.avancar(0.0, espaco=True)
    sem_salto.mundo.t_jogo = t_ini
    sem_salto.mundo.esferas.append(mundo.Esfera(t_spawn=0.0, fase=fase))
    for _ in range(passagens):
        sem_salto.avancar(passo)
    return validos * passo, sem_salto.estado == "fim"


@pytest.mark.parametrize("fase", [0.0, math.pi / 4, math.pi / 2, 3 * math.pi / 4])
def test_ha_janela_de_acerto_de_pelo_menos_200_ms_para_qualquer_quicada(fase):
    janela, colide_sem_pular = janela_de_acerto(fase)
    assert colide_sem_pular
    assert janela >= 0.2 - 1e-9


# Acrescente 2 a 3 casos seus: (centro da esfera (x, y, z), esperado: True se colide) com o cubo de lado 1.2
# centrado em (0, 0.6, 0) e a esfera de raio 0.6.
CASOS_DO_USUARIO = []


@pytest.mark.parametrize("centro_esfera, esperado", CASOS_DO_USUARIO)
def test_casos_do_usuario(centro_esfera, esperado):
    assert jogo.colisao_cubo_esfera(CUBO, 1.2, centro_esfera, 0.6) == esperado
