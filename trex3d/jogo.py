"""Mecanica do jogo: estados, colisao cubo x esfera e pontuacao.

Estados: "esperando" (espaco inicia), "jogando" (espaco pula; colisao leva ao fim) e "fim" (o jogo
congela e o Sol continua; espaco reinicia depois de um pequeno atraso, para nao reiniciar sem querer).
A colisao e testada a cada sub-passo de no maximo PASSO_MAXIMO, entao um quadro longo nao atravessa o cubo.
"""
import math

from trex3d import mundo

ESPERANDO, JOGANDO, FIM = "esperando", "jogando", "fim"
PASSO_MAXIMO = 1.0 / 120.0

# Salto mais generoso que o padrao de mundo.Parametros (2.0 m em 0.8 s): medido em tests/test_jogo.py,
# a janela de tempo que ainda passa por cima de uma esfera quicando sobe de ~146 ms para ~280 ms.
PARAMETROS_DO_JOGO = mundo.Parametros(salto_altura=2.4, salto_duracao=0.9)


def colisao_cubo_esfera(centro_cubo, lado, centro_esfera, raio):
    """Ha colisao se o ponto do cubo mais proximo do centro da esfera esta a menos de `raio` dele."""
    meio = lado / 2.0
    d2 = 0.0
    for c_cubo, c_esfera in zip(centro_cubo, centro_esfera):
        folga = max(abs(c_esfera - c_cubo) - meio, 0.0)
        d2 += folga * folga
    return d2 < raio * raio


def piloto_automatico(j):
    """True se deve pular agora: a proxima esfera esta a meio tempo de salto do cubo, de modo que o
    topo do salto cobre o encontro. Usado na demonstracao (main.py --piloto) e nos testes."""
    m = j.mundo
    if m.no_ar:
        return False
    alcance = m.p.velocidade * m.p.salto_duracao / 2.0
    return any(m.p.cubo_x < m.estado_da_esfera(e)[0] <= m.p.cubo_x + alcance for e in m.esferas)


class Jogo:
    def __init__(self, parametros=PARAMETROS_DO_JOGO, atraso_reinicio=0.5):
        self.mundo = mundo.Mundo(parametros)
        self.atraso_reinicio = atraso_reinicio
        self.estado = ESPERANDO
        self.pontos = 0
        self.recorde = 0
        self.esfera_colidida = None
        self._contadas = set()
        self._tempo_em_fim = 0.0

    @property
    def tempo(self):
        return self.mundo.t_jogo

    def avancar(self, dt, espaco=False):
        if dt < 0:
            raise ValueError("dt nao pode ser negativo")
        self.mundo.avancar_sol(dt)
        if self.estado == ESPERANDO:
            if espaco:
                self._iniciar()
        elif self.estado == JOGANDO:
            self._avancar_partida(dt, espaco)
        else:
            self._tempo_em_fim += dt
            if espaco and self._tempo_em_fim >= self.atraso_reinicio:
                self._iniciar()

    def _iniciar(self):
        self.mundo.reiniciar_jogo()
        self.pontos = 0
        self.esfera_colidida = None
        self._contadas.clear()
        self.estado = JOGANDO

    def _avancar_partida(self, dt, espaco):
        n = max(1, math.ceil(dt / PASSO_MAXIMO - 1e-9))
        h = dt / n
        for i in range(n):
            self.mundo.avancar_jogo(h, pular=espaco and i == 0)
            self._contar_ultrapassadas()
            if self._colidiu():
                self._terminar()
                return

    def _contar_ultrapassadas(self):
        p = self.mundo.p
        limite = p.cubo_x - p.cubo_lado / 2.0 - p.esfera_raio
        for e in self.mundo.esferas:
            if e.t_spawn not in self._contadas and self.mundo.estado_da_esfera(e)[0] < limite:
                self._contadas.add(e.t_spawn)
                self.pontos += 1

    def _colidiu(self):
        p = self.mundo.p
        cubo = (p.cubo_x, self.mundo.cubo_y, 0.0)
        for e in self.mundo.esferas:
            x, y, _ = self.mundo.estado_da_esfera(e)
            if colisao_cubo_esfera(cubo, p.cubo_lado, (x, y, 0.0), p.esfera_raio):
                self.esfera_colidida = e
                return True
        return False

    def _terminar(self):
        self.estado = FIM
        self._tempo_em_fim = 0.0
        self.recorde = max(self.recorde, self.pontos)
