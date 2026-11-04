"""T-Rex Game 3D (Computacao Grafica, UNIOESTE 2026).

Barra de espaco: inicia, pula e reinicia. 1, 2, 3: sombreamento constante, Gouraud, Phong.
T (segurar): acelera o Sol. Esc: sai. As opcoes da linha de comando estao no README.
"""
import argparse
import sys
import time

import numpy as np
import pygame

from trex3d import cena as cena_mod
from trex3d import jogo as jogo_mod
from trex3d import sombreamento

TECLAS_MODO = {pygame.K_1: "constante", pygame.K_2: "gouraud", pygame.K_3: "phong"}
FATOR_DA_TECLA_T = 20.0
COR_TEXTO = (40, 40, 40)


def escrever(tela, fonte, texto, pos, centro=False, direita=False):
    img = fonte.render(texto, True, COR_TEXTO)
    r = img.get_rect()
    if centro:
        r.center = pos
    elif direita:
        r.topright = pos
    else:
        r.topleft = pos
    tela.blit(img, r)


def desenhar_hud(tela, fontes, jogo, modo, fps):
    grande, media, pequena = fontes
    escrever(tela, media, f"HI {jogo.recorde:05d}   {jogo.pontos:05d}", (cena_mod.LARGURA - 15, 12), direita=True)
    escrever(tela, pequena, f"Sombreamento: {modo} (teclas 1, 2, 3)   T: acelera o Sol   {fps:.0f} FPS",
             (12, cena_mod.ALTURA - 24))
    if jogo.estado == jogo_mod.ESPERANDO:
        escrever(tela, grande, "Pressione a barra de espaço para jogar", (cena_mod.LARGURA // 2, 120), centro=True)
    elif jogo.estado == jogo_mod.FIM:
        escrever(tela, grande, "Fim de jogo", (cena_mod.LARGURA // 2, 110), centro=True)
        escrever(tela, media, "Barra de espaço reinicia", (cena_mod.LARGURA // 2, 150), centro=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description="T-Rex Game 3D")
    ap.add_argument("--modo", choices=sombreamento.MODOS, default="phong")
    ap.add_argument("--piloto", action="store_true")
    ap.add_argument("--fator-sol", type=float, default=1.0)
    ap.add_argument("--quadros", type=int, default=None)
    ap.add_argument("--captura", default=None)
    args = ap.parse_args(argv)
    if args.fator_sol < 1.0:
        ap.error("--fator-sol deve ser >= 1")

    pygame.init()
    tela = pygame.display.set_mode((cena_mod.LARGURA, cena_mod.ALTURA))
    pygame.display.set_caption("T-Rex Game 3D - Computação Gráfica")
    fontes = (pygame.font.Font(None, 44), pygame.font.Font(None, 32), pygame.font.Font(None, 22))

    jogo = jogo_mod.Jogo()
    cena = cena_mod.Cena(jogo)
    fb = np.zeros((cena_mod.LARGURA, cena_mod.ALTURA, 3), np.uint8)
    relogio = pygame.time.Clock()
    modo = args.modo
    quadros = 0
    inicio = time.perf_counter()
    rodando = True
    while rodando:
        if args.quadros is None:
            dt = min(relogio.tick(60) / 1000.0, 0.05)
            fps = relogio.get_fps()
        else:
            dt = 1.0 / 60.0
            fps = quadros / max(time.perf_counter() - inicio, 1e-9)
        espaco = False
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                rodando = False
            elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_SPACE:
                espaco = True
            elif ev.type == pygame.KEYDOWN and ev.key in TECLAS_MODO:
                modo = TECLAS_MODO[ev.key]
        if args.piloto:
            espaco = espaco or jogo.estado != jogo_mod.JOGANDO or jogo_mod.piloto_automatico(jogo)

        jogo.avancar(dt, espaco)
        fator_sol = args.fator_sol * (FATOR_DA_TECLA_T if pygame.key.get_pressed()[pygame.K_t] else 1.0)
        if fator_sol > 1.0:
            jogo.mundo.avancar_sol(dt * (fator_sol - 1.0))
        cena.desenhar(fb, modo)
        pygame.surfarray.blit_array(tela, fb)
        desenhar_hud(tela, fontes, jogo, modo, fps)
        pygame.display.flip()

        quadros += 1
        if args.quadros is not None and quadros >= args.quadros:
            rodando = False

    if args.captura:
        pygame.image.save(tela, args.captura)
    if args.quadros is not None:
        total = time.perf_counter() - inicio
        print(f"{quadros} quadros em {total:.1f} s = {quadros / total:.0f} FPS | estado={jogo.estado} "
              f"pontos={jogo.pontos} recorde={jogo.recorde} tempo_de_jogo={jogo.tempo:.1f} s | modo={modo} | "
              f"sol_x={jogo.mundo.posicao_do_sol()[0]:.2f}")
    pygame.quit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
