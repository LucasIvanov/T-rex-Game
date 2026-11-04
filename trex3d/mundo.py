"""Mundo e animacao (parte 8): cubo que salta, esferas que rolam e quicam, Sol lento.

Posicao e orientacao sao funcoes do tempo (salto parabolico, x = x0 - v t, angulo = distancia / raio)
e cada esfera nasce no instante exato programado, nao no quadro em que foi percebida. Por isso
passos de 1/30 s, 1/120 s ou irregulares levam ao mesmo estado.

Dois relogios: o do Sol (sempre anda) e o do jogo (esferas e salto; parado fora da partida).
Eixos: x para a direita, y para cima, z para o observador.
"""
import math
import random
from dataclasses import dataclass

import numpy as np

from trex3d import mat

TOLERANCIA_TEMPO = 1e-9


@dataclass(frozen=True)
class Parametros:
    cubo_lado: float = 1.2
    cubo_x: float = -5.5
    salto_altura: float = 2.0
    salto_duracao: float = 0.8
    esfera_raio: float = 0.6
    velocidade: float = 6.0
    spawn_x: float = 8.0
    remocao_x: float = -8.5
    atraso_inicial: float = 1.0
    folga_intervalo: float = 0.35
    variacao_intervalo: float = 1.0
    quicada_altura: float = 0.3
    quicada_periodo: float = 0.8
    quicadas: bool = True
    rolagem: bool = True
    sol_x_inicial: float = 6.0
    sol_velocidade: float = 0.2
    sol_limite: float = 8.0
    sol_y_base: float = 6.5
    sol_y_arco: float = 1.0
    sol_z: float = 8.0
    semente: int = 0

    def __post_init__(self):
        for nome in ("cubo_lado", "salto_altura", "salto_duracao", "esfera_raio", "velocidade",
                     "quicada_periodo", "sol_limite"):
            if getattr(self, nome) <= 0:
                raise ValueError(f"{nome} deve ser positivo")
        for nome in ("sol_velocidade", "atraso_inicial", "folga_intervalo", "variacao_intervalo", "quicada_altura"):
            if getattr(self, nome) < 0:
                raise ValueError(f"{nome} nao pode ser negativo")

    @property
    def v0(self):
        """Velocidade inicial do salto: 4 H / T (topo H em T/2)."""
        return 4.0 * self.salto_altura / self.salto_duracao

    @property
    def g(self):
        """Gravidade do salto: 8 H / T^2."""
        return 8.0 * self.salto_altura / self.salto_duracao ** 2

    @property
    def intervalo_min(self):
        """Tempo minimo entre esferas: o tempo no ar do salto mais uma folga para o jogador reagir."""
        return self.salto_duracao + self.folga_intervalo

    @property
    def intervalo_max(self):
        return self.intervalo_min + self.variacao_intervalo


@dataclass
class Esfera:
    t_spawn: float
    fase: float


class Mundo:
    def __init__(self, parametros=None):
        self.p = parametros or Parametros()
        self._rng = random.Random(self.p.semente)
        self.t_sol = 0.0
        self.reiniciar_jogo()

    def reiniciar_jogo(self):
        """Limpa o jogo (esferas, salto e relogio do jogo); o Sol continua de onde estava."""
        self.t_jogo = 0.0
        self.esferas = []
        self.historico_spawns = []
        self.no_ar = False
        self._ts = 0.0
        self._proximo_spawn = self.p.atraso_inicial

    def avancar(self, dt, pular=False):
        self.avancar_sol(dt)
        self.avancar_jogo(dt, pular)

    def avancar_sol(self, dt):
        _validar_dt(dt)
        self.t_sol += dt

    def avancar_jogo(self, dt, pular=False):
        _validar_dt(dt)
        p = self.p
        if pular and not self.no_ar:
            self.no_ar = True
            self._ts = 0.0
        if self.no_ar:
            self._ts += dt
            if self._ts >= p.salto_duracao - TOLERANCIA_TEMPO:
                self.no_ar = False
                self._ts = 0.0
        self.t_jogo += dt
        while self._proximo_spawn <= self.t_jogo + 1e-12:
            self.esferas.append(Esfera(self._proximo_spawn, self._rng.uniform(0.0, 2 * math.pi)))
            self.historico_spawns.append(self._proximo_spawn)
            self._proximo_spawn += self._rng.uniform(p.intervalo_min, p.intervalo_max)
        self.esferas = [e for e in self.esferas if self.estado_da_esfera(e)[0] >= p.remocao_x]

    @property
    def cubo_y(self):
        """Altura do centro do cubo."""
        p = self.p
        base = p.cubo_lado / 2.0
        if not self.no_ar:
            return base
        return base + max(0.0, p.v0 * self._ts - 0.5 * p.g * self._ts ** 2)

    def matriz_cubo(self):
        return mat.translacao(self.p.cubo_x, self.cubo_y, 0.0)

    def estado_da_esfera(self, e):
        """(x, y, angulo) do centro e da rotacao em torno de z (rolando para a esquerda: anti-horario)."""
        p = self.p
        tl = self.t_jogo - e.t_spawn
        x = p.spawn_x - p.velocidade * tl
        y = p.esfera_raio
        if p.quicadas:
            y += p.quicada_altura * abs(math.sin(math.pi * tl / p.quicada_periodo + e.fase))
        theta = p.velocidade * tl / p.esfera_raio if p.rolagem else 0.0
        return x, y, theta

    def matrizes_esferas(self):
        matrizes = []
        for e in self.esferas:
            x, y, theta = self.estado_da_esfera(e)
            matrizes.append(mat.compor(mat.translacao(x, y, 0.0), mat.rot_z(theta)))
        return matrizes

    def posicao_do_sol(self):
        """Sol da direita para a esquerda em arco suave; ao sair pela esquerda volta pela direita."""
        p = self.p
        lim = p.sol_limite
        x = ((p.sol_x_inicial - p.sol_velocidade * self.t_sol + lim) % (2 * lim)) - lim
        y = p.sol_y_base + p.sol_y_arco * math.cos(math.pi * x / (2 * lim))
        return np.array([x, y, p.sol_z])


def _validar_dt(dt):
    if dt < 0:
        raise ValueError("dt nao pode ser negativo")
