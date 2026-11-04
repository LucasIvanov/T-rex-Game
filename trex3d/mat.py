"""Nucleo matematico: vetores 3D e matrizes 4x4 homogeneas.

Convencoes (apostila, Un. 4): vetor coluna, P' = M * P; mao direita; angulos em radianos,
positivos no sentido anti-horario. Em compor(A, B), B e aplicada primeiro.
"""
import math

import numpy as np


def escalar(a, b):
    return float(a[0] * b[0] + a[1] * b[1] + a[2] * b[2])


def vetorial(a, b):
    return np.array([
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ], dtype=float)


def normalizar(v):
    comprimento = math.sqrt(escalar(v, v))
    if comprimento == 0:
        raise ValueError("vetor nulo nao pode ser normalizado")
    return np.asarray(v, dtype=float) / comprimento


def translacao(dx, dy, dz):
    m = np.eye(4)
    m[:3, 3] = (dx, dy, dz)
    return m


def escala(sx, sy, sz):
    return np.diag([sx, sy, sz, 1.0])


def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0, 0],
                     [0, c, -s, 0],
                     [0, s, c, 0],
                     [0, 0, 0, 1]], dtype=float)


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s, 0],
                     [0, 1, 0, 0],
                     [-s, 0, c, 0],
                     [0, 0, 0, 1]], dtype=float)


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0, 0],
                     [s, c, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]], dtype=float)


def compor(*matrizes):
    resultado = np.eye(4)
    for m in matrizes:
        resultado = resultado @ m
    return resultado


def em_torno_de(m, ponto):
    """Aplica m com o ponto como origem: T(ponto) * m * T(-ponto)."""
    px, py, pz = ponto
    return compor(translacao(px, py, pz), m, translacao(-px, -py, -pz))


def so_rotacao_e_escala_uniforme(m):
    """True se a parte 3x3 de m e uma rotacao vezes uma escala uniforme (conserva angulos; normais so giram)."""
    m3 = np.asarray(m, dtype=float)[:3, :3]
    gram = m3.T @ m3
    return bool(np.allclose(gram, gram[0, 0] * np.eye(3), rtol=1e-9, atol=1e-12))


def aplicar(m, pontos):
    """Transforma um ponto (3,) ou uma lista de pontos (N, 3) e divide pelo fator homogeneo."""
    p = np.asarray(pontos, dtype=float)
    xyz = p @ m[:3, :3].T + m[:3, 3]
    w = np.asarray(p @ m[3, :3] + m[3, 3])
    if np.any(w == 0):
        raise ValueError("fator homogeneo nulo (ponto no infinito)")
    return xyz / w[..., None]
