"""Modelos poliedricos (apostila Un. 3): vertices em lista unica e faces como indices (3.2.2).

Convencoes: objetos centrados na origem; cada face lista seus vertices em sentido anti-horario
visto de fora (normal para fora); triangulos sao completados com o ultimo vertice repetido
para que toda face tenha 4 indices.
"""
import math

import numpy as np

from trex3d import mat


def normais_das_faces(pontos):
    """Normais unitarias de faces (F, K, 3): soma dos produtos vetoriais de vertices sucessivos
    (Gouraud 1971, p. 29). Vale para qualquer poligono plano e ignora vertices repetidos."""
    p = np.asarray(pontos, dtype=float)
    soma = np.cross(p, np.roll(p, -1, axis=1)).sum(axis=1)
    comprimento = np.linalg.norm(soma, axis=1, keepdims=True)
    if np.any(comprimento < 1e-12):
        raise ValueError("face degenerada (area nula)")
    return soma / comprimento


def normal_da_face(vertices):
    return normais_das_faces(np.asarray(vertices, dtype=float)[None])[0]


class Malha:
    def __init__(self, vertices, normais_vertice, faces):
        self.vertices = np.asarray(vertices, dtype=float)
        self.normais_vertice = np.asarray(normais_vertice, dtype=float)
        self.faces = np.asarray(faces, dtype=np.int64)
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3:
            raise ValueError("vertices devem ter forma (V, 3)")
        if self.normais_vertice.shape != self.vertices.shape:
            raise ValueError("uma normal por vertice")
        if self.faces.ndim != 2 or self.faces.shape[1] != 4:
            raise ValueError("faces devem ter forma (F, 4)")
        if self.faces.min() < 0 or self.faces.max() >= self.vertices.shape[0]:
            raise ValueError("indice de vertice fora da lista")
        self.normais_face = normais_das_faces(self.vertices[self.faces])

    @property
    def centros_das_faces(self):
        """Media dos vertices distintos de cada face (um triangulo tem 3, nao 4)."""
        p = self.vertices[self.faces]
        quadrilatero = (self.faces[:, 3] != self.faces[:, 2])[:, None]
        soma = p[:, 0] + p[:, 1] + p[:, 2] + np.where(quadrilatero, p[:, 3], 0.0)
        return soma / np.where(quadrilatero, 4.0, 3.0)


def criar_caixa(lx, ly, lz):
    """Caixa lx x ly x lz com 24 vertices (4 por face, para manter as arestas vivas no Gouraud e no Phong)."""
    if min(lx, ly, lz) <= 0:
        raise ValueError("os lados da caixa devem ser positivos")
    h = (lx / 2.0, ly / 2.0, lz / 2.0)
    vertices, normais, faces = [], [], []
    for eixo in range(3):
        u, v = (eixo + 1) % 3, (eixo + 2) % 3   # u x v = eixo (mao direita)
        for sinal in (1, -1):
            cantos = [(-1, -1), (1, -1), (1, 1), (-1, 1)]   # anti-horario visto de +eixo
            if sinal < 0:
                cantos.reverse()
            normal = np.zeros(3)
            normal[eixo] = sinal
            base = len(vertices)
            for du, dv in cantos:
                p = np.zeros(3)
                p[eixo] = sinal * h[eixo]
                p[u] = du * h[u]
                p[v] = dv * h[v]
                vertices.append(p)
                normais.append(normal)
            faces.append([base, base + 1, base + 2, base + 3])
    return Malha(vertices, normais, faces)


def criar_cubo(lado):
    return criar_caixa(lado, lado, lado)


def combinar(partes):
    """Junta malhas em uma so: partes = [(malha, matriz 4x4), ...]. As matrizes so podem ter rotacao,
    translacao e escala uniforme (as normais giram e sao renormalizadas)."""
    vertices, normais, faces, base = [], [], [], 0
    for malha, m in partes:
        if not mat.so_rotacao_e_escala_uniforme(m):
            raise ValueError("escala nao uniforme: as normais dos vertices ficariam incorretas")
        n = malha.normais_vertice @ np.asarray(m, dtype=float)[:3, :3].T
        normais.append(n / np.linalg.norm(n, axis=1, keepdims=True))
        vertices.append(mat.aplicar(m, malha.vertices))
        faces.append(malha.faces + base)
        base += malha.vertices.shape[0]
    return Malha(np.concatenate(vertices), np.concatenate(normais), np.concatenate(faces))


def criar_cacto():
    """Cacto de 5 caixas (tronco e dois bracos em L), apoiado em y = 0, com 2.0 de altura e 1.2 de largura."""
    tronco = criar_caixa(0.35, 2.0, 0.35)
    braco_horizontal = criar_caixa(0.425, 0.25, 0.25)
    braco_vertical = criar_caixa(0.25, 0.7, 0.25)
    partes = [(tronco, mat.translacao(0.0, 1.0, 0.0))]
    for lado in (-1, 1):
        partes.append((braco_horizontal, mat.translacao(lado * 0.3875, 0.9, 0.0)))
        partes.append((braco_vertical, mat.translacao(lado * 0.475, 1.125, 0.0)))
    return combinar(partes)


def criar_esfera(raio, faixas, fatias):
    """Esfera por faixas (paralelos) e fatias (meridianos), com um vertice em cada polo.

    Vertices: 0 = polo norte, (faixas - 1) aneis de `fatias` vertices, ultimo = polo sul.
    Faces: as do polo norte, as quadrilateras de cima para baixo, as do polo sul.
    """
    if raio <= 0 or faixas < 2 or fatias < 3:
        raise ValueError("esfera precisa de raio > 0, ao menos 2 faixas e 3 fatias")
    vertices = [[0.0, raio, 0.0]]
    for s in range(1, faixas):
        phi = math.pi * s / faixas
        for j in range(fatias):
            theta = 2 * math.pi * j / fatias
            vertices.append([raio * math.sin(phi) * math.cos(theta),
                             raio * math.cos(phi),
                             raio * math.sin(phi) * math.sin(theta)])
    vertices.append([0.0, -raio, 0.0])
    vertices = np.array(vertices)
    ultimo = len(vertices) - 1

    def anel(s, j):
        return 1 + (s - 1) * fatias + (j % fatias)

    faces = []
    for j in range(fatias):
        faces.append([0, anel(1, j + 1), anel(1, j), anel(1, j)])
    for s in range(1, faixas - 1):
        for j in range(fatias):
            faces.append([anel(s, j), anel(s, j + 1), anel(s + 1, j + 1), anel(s + 1, j)])
    for j in range(fatias):
        faces.append([ultimo, anel(faixas - 1, j), anel(faixas - 1, j + 1), anel(faixas - 1, j + 1)])
    return Malha(vertices, vertices / raio, faces)


def criar_plano(x0, x1, z0, z1, y=0.0):
    """Retangulo horizontal na altura y, com a normal para +y."""
    vertices = [[x0, y, z0], [x0, y, z1], [x1, y, z1], [x1, y, z0]]
    return Malha(vertices, [[0, 1, 0]] * 4, [[0, 1, 2, 3]])
