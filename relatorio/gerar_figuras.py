"""Gera as figuras do relatorio em relatorio/figuras/ (sem janela).

Uso, na raiz do projeto: python relatorio/gerar_figuras.py
"""
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

import numpy as np
import pygame

import main
from trex3d import cena as cena_mod
from trex3d import jogo as jogo_mod

SAIDA = RAIZ / "relatorio" / "figuras"
FUNDO = (255, 255, 255)


def partida(segundos, piloto=False):
    j = jogo_mod.Jogo()
    j.avancar(0.0, espaco=True)
    for _ in range(int(segundos * 60)):
        j.avancar(1 / 60, espaco=piloto and jogo_mod.piloto_automatico(j))
    return j


def quadro(j, modo):
    fb = np.zeros((cena_mod.LARGURA, cena_mod.ALTURA, 3), np.uint8)
    cena_mod.Cena(j).desenhar(fb, modo)
    return fb


def rotulo(texto, largura, fonte, altura=30):
    s = pygame.Surface((largura, altura))
    s.fill(FUNDO)
    img = fonte.render(texto, True, (30, 30, 30))
    s.blit(img, img.get_rect(center=(largura // 2, altura // 2)))
    return pygame.surfarray.array3d(s)


def salvar(arr, nome):
    SAIDA.mkdir(parents=True, exist_ok=True)
    pygame.image.save(pygame.surfarray.make_surface(arr), str(SAIDA / nome))
    print("gerada", nome, f"{arr.shape[0]}x{arr.shape[1]}")


def com_hud(fb, j, modo, fontes, fps):
    tela = pygame.display.get_surface()
    pygame.surfarray.blit_array(tela, fb)
    main.desenhar_hud(tela, fontes, j, modo, fps)
    return pygame.surfarray.array3d(tela)


def main_figuras():
    pygame.init()
    pygame.display.set_mode((cena_mod.LARGURA, cena_mod.ALTURA))
    fontes = (pygame.font.Font(None, 44), pygame.font.Font(None, 32), pygame.font.Font(None, 22))
    legenda = pygame.font.Font(None, 26)

    # fig1: a mesma esfera nos tres sombreamentos (recorte de 80 x 80 px ampliado 4 vezes)
    j = partida(2.0)
    c = cena_mod.Cena(j)
    x, y, _ = j.mundo.estado_da_esfera(j.mundo.esferas[0])
    px, py = (int(round(v)) for v in c.camera.projetar((x, y, 0))[:2])
    pedacos = []
    for modo, nome in (("constante", "Constante"), ("gouraud", "Gouraud"), ("phong", "Phong")):
        corte = quadro(j, modo)[px - 40:px + 40, py - 40:py + 40]
        ampliado = np.repeat(np.repeat(corte, 4, axis=0), 4, axis=1)
        pedacos.append(np.concatenate([rotulo(nome, 320, legenda), ampliado], axis=1))
    espaco = np.full((12, pedacos[0].shape[1], 3), 255, np.uint8)
    salvar(np.concatenate([pedacos[0], espaco, pedacos[1], espaco, pedacos[2]], axis=0), "fig1_sombreamentos.png")

    # fig2: o Sol em tres instantes (mesmo mundo; tempo do Sol 0, 30 e 60 s), a metade do tamanho
    faixas = []
    for t_sol in (0.0, 30.0, 60.0):
        j = partida(2.0)
        j.mundo.avancar_sol(t_sol)
        sx = j.mundo.posicao_do_sol()[0]
        meio = quadro(j, "phong")[::2, ::2]
        faixas.append(np.concatenate([rotulo(f"Sol em x = {sx:.1f}  (t = {j.mundo.t_sol:.0f} s)", meio.shape[0], legenda), meio], axis=1))
    espaco = np.full((6, faixas[0].shape[1], 3), 255, np.uint8)
    salvar(np.concatenate([faixas[0], espaco, faixas[1], espaco, faixas[2]], axis=0), "fig2_sol.png")

    # fig3: partida em andamento com o piloto automatico (FPS medido aqui, em Phong)
    j = partida(0.0)
    inicio = time.perf_counter()
    quadros = 0
    for _ in range(60 * 14):
        j.avancar(1 / 60, espaco=jogo_mod.piloto_automatico(j))
        fb = quadro(j, "phong")
        quadros += 1
    fps = quadros / (time.perf_counter() - inicio)
    salvar(com_hud(fb, j, "phong", fontes, fps), "fig3_partida.png")

    # fig4: fim de jogo
    j = partida(0.0)
    while j.estado == jogo_mod.JOGANDO:
        j.avancar(1 / 60)
    for _ in range(12):
        j.avancar(1 / 60)
    salvar(com_hud(quadro(j, "phong"), j, "phong", fontes, fps), "fig4_fim_de_jogo.png")
    pygame.quit()


if __name__ == "__main__":
    main_figuras()
