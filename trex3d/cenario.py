"""Cenario de fundo: nuvens fixas na tela (3 elipses cada, pintadas depois do Sol, que passa por
tras delas) e cactos decorativos atras da pista (z = 0), cada um com sua escala uniforme.
"""
import math

import numpy as np

COR_NUVEM = (245, 248, 252)

# centro na tela (px, py) e escala da nuvem; todas acima do horizonte do terreno (py = 187)
NUVENS = [(140, 62, 1.0), (360, 40, 0.8), (690, 70, 1.2)]
# elipses de uma nuvem: (deslocamento x, deslocamento y, raio x, raio y); a primeira e a base
FORMA_DA_NUVEM = [(0, 0, 60, 14), (-22, -10, 30, 16), (20, -14, 26, 16)]

# (x, z, escala): cinco tamanhos diferentes, de 0.6 a 1.4 vezes o cacto de referencia
CACTOS_DE_FUNDO = [
    (-6.0, -3.0, 0.8),
    (-2.5, -4.5, 1.4),
    (1.0, -3.5, 0.6),
    (4.2, -5.0, 1.1),
    (6.8, -3.0, 0.9),
]


def elipse(cx, cy, rx, ry, n=24):
    t = 2 * math.pi * np.arange(n) / n
    return np.stack([cx + rx * np.cos(t), cy + ry * np.sin(t)], axis=-1)


def poligonos_das_nuvens():
    """(P, cores): P (N, 24, 2) vertices em pixels e cores (N, 3) uint8; 3 poligonos por nuvem."""
    poligonos = []
    for cx, cy, s in NUVENS:
        for dx, dy, rx, ry in FORMA_DA_NUVEM:
            poligonos.append(elipse(cx + s * dx, cy + s * dy, s * rx, s * ry))
    P = np.array(poligonos)
    return P, np.tile(np.array(COR_NUVEM, dtype=np.uint8), (P.shape[0], 1))
