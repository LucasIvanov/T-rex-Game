"""Spike (descartavel): janela pygame + numpy para testar o empacotamento com PyInstaller.

Uso: hello [--auto-quit]   (--auto-quit fecha sozinho depois de ~2 s, para teste automatico)
Grava hello.log ao lado do executavel (ou na pasta temporaria se ela nao for gravavel).
"""
import os
import sys
import time
import tempfile
import traceback

T0 = time.perf_counter()


def pasta_do_app():
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def abrir_log():
    for pasta in (pasta_do_app(), tempfile.gettempdir()):
        try:
            caminho = os.path.join(pasta, "hello.log")
            return open(caminho, "a", encoding="utf-8"), caminho
        except OSError:
            continue
    return None, None


def main():
    log, caminho = abrir_log()

    def escrever(msg):
        if log:
            log.write(msg + "\n")
            log.flush()
        print(msg)

    try:
        import numpy as np
        import pygame

        escrever("--- execucao " + time.strftime("%Y-%m-%d %H:%M:%S"))
        escrever(f"python={sys.version.split()[0]} pygame={pygame.version.ver} numpy={np.__version__}")
        escrever(f"frozen={getattr(sys, 'frozen', False)} executavel={sys.executable}")
        escrever(f"MEIPASS={getattr(sys, '_MEIPASS', '-')} cwd={os.getcwd()} log={caminho}")
        escrever(f"imports prontos em {(time.perf_counter() - T0) * 1000:.0f} ms")

        pygame.init()
        w, h = 640, 360
        tela = pygame.display.set_mode((w, h))
        pygame.display.set_caption("hello_exe - spike")
        quadro = np.zeros((w, h, 3), np.uint8)
        quadro[..., 0] = np.linspace(0, 255, w, dtype=np.uint8)[:, None]
        quadro[..., 1] = np.linspace(0, 255, h, dtype=np.uint8)[None, :]
        quadro[..., 2] = 128
        escrever(f"janela aberta em {(time.perf_counter() - T0) * 1000:.0f} ms")

        auto = "--auto-quit" in sys.argv
        relogio = pygame.time.Clock()
        n = 0
        rodando = True
        while rodando:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                    rodando = False
            quadro[..., 2] = (n * 4) % 256
            pygame.surfarray.blit_array(tela, quadro)
            pygame.display.flip()
            relogio.tick(60)
            n += 1
            if auto and time.perf_counter() - T0 > 3.0:
                rodando = False
        escrever(f"{n} quadros; fim ok")
        pygame.quit()
    except Exception:
        escrever("ERRO:\n" + traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
