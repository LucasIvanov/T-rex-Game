"""Montagem do quadro.

Ordem de pintura: ceu, Sol, nuvens, terreno, cactos de fundo e, por fim, cubo e esferas do mais longe
ao mais perto. Terreno e cactos usam sombreamento constante (o Phong na maior area da tela custava
cerca de 3 vezes mais por quadro); cubo e esferas usam o modo escolhido. O Sol desenhado e a mesma
posicao usada como fonte de luz.
"""
import math

import numpy as np

from trex3d import camera, cenario, luz, mat, modelos, raster, sombreamento, visibilidade

LARGURA, ALTURA = 800, 450
ESCALA = 55.0                       # pixels por unidade do mundo
ELEVACAO = math.radians(20)
COR_CEU = (120, 180, 235)
MATERIAIS = [luz.MATERIAL_TERRENO, luz.MATERIAL_CUBO, luz.MATERIAL_ESFERA, luz.MATERIAL_SOL, luz.MATERIAL_CACTO]
ID_TERRENO, ID_CUBO, ID_ESFERA, ID_SOL, ID_CACTO = range(5)


class Cena:
    def __init__(self, jogo):
        self.jogo = jogo
        p = jogo.mundo.p
        n = (0.0, math.sin(ELEVACAO), math.cos(ELEVACAO))
        janela = (-LARGURA / 2 / ESCALA, -150 / ESCALA, LARGURA / 2 / ESCALA, 300 / ESCALA)
        self.camera = camera.CameraOrtografica((0, 0, 0), n, (0, 1, 0), janela, (0, 0, LARGURA, ALTURA))
        self.terreno = modelos.criar_plano(-8, 8, -6, 8)
        self.cubo = modelos.criar_cubo(p.cubo_lado)
        self.esfera = modelos.criar_esfera(p.esfera_raio, 6, 8)
        self.disco_do_sol = modelos.criar_esfera(0.5, 12, 16)
        self.cacto = modelos.criar_cacto()
        self.nuvens, self.cores_das_nuvens = cenario.poligonos_das_nuvens()

    def _pintar(self, fb, modo, itens, fonte):
        faces = sombreamento.juntar([sombreamento.preparar_faces(m, mm, self.camera, i) for m, mm, i in itens])
        return sombreamento.pintar_faces(fb, modo, faces, MATERIAIS, fonte, self.camera.n)

    def _do_fundo_para_a_frente(self, objetos):
        centros = np.array([visibilidade.centro_do_objeto(m, mm) for m, mm, _ in objetos])
        return [objetos[i] for i in visibilidade.ordem_do_pintor(centros, self.camera)]

    def desenhar(self, fb, modo):
        """Pinta o quadro em fb (LARGURA, ALTURA, 3) uint8."""
        mundo = self.jogo.mundo
        sol = mundo.posicao_do_sol()
        fonte = luz.Luz(posicao=sol)
        fb[:] = COR_CEU
        self._pintar(fb, "constante", [(self.disco_do_sol, mat.translacao(*sol), ID_SOL)], fonte)
        raster.preencher(fb, self.nuvens, self.cores_das_nuvens)
        self._pintar(fb, "constante", [(self.terreno, np.eye(4), ID_TERRENO)], fonte)
        cactos = [(self.cacto, mat.compor(mat.translacao(x, 0.0, z), mat.escala(s, s, s)), ID_CACTO)
                  for x, z, s in cenario.CACTOS_DE_FUNDO]
        self._pintar(fb, "constante", self._do_fundo_para_a_frente(cactos), fonte)
        objetos = [(self.cubo, mundo.matriz_cubo(), ID_CUBO)]
        objetos += [(self.esfera, m, ID_ESFERA) for m in mundo.matrizes_esferas()]
        self._pintar(fb, modo, self._do_fundo_para_a_frente(objetos), fonte)
