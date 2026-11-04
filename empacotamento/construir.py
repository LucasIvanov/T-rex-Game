"""Gera os executaveis do T-Rex Game 3D com PyInstaller e confere cada um com um teste de fumaca.

Uso, na raiz do projeto: python empacotamento/construir.py [--so-pasta | --so-unico]

Saidas em _saida/: dist/TRexGame3D/ (pasta, a forma principal), dist/TRexGame3D_unico.exe e, em
entrega/, o zip da pasta e uma copia do arquivo unico.

O teste de fumaca roda 180 quadros com o piloto, sem janela (SDL_VIDEODRIVER=dummy), a partir de uma
pasta com espaco e acento no nome, e confere o codigo de saida, o resumo e a captura de tela. O
executavel mantem o console de proposito: se falhar em outra maquina, o erro aparece nele.
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
SAIDA = RAIZ / "_saida"
NOME_PASTA = "TRexGame3D"
NOME_UNICO = "TRexGame3D_unico"
PASTA_DE_TESTE = "pasta com espaço e acentuação ção"
QUADROS = 180


def pyinstaller(modo, nome):
    comando = [sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", modo, "--name", nome,
               "--distpath", str(SAIDA / "dist"), "--workpath", str(SAIDA / "build"),
               "--specpath", str(SAIDA / "spec"), str(RAIZ / "main.py")]
    print(f"\n== PyInstaller {modo} -> {nome}")
    subprocess.run(comando, cwd=RAIZ, check=True)


def fumaca(exe, rotulo):
    """Roda o executavel sem janela e confere. Devolve uma lista de problemas (vazia = ok)."""
    problemas = []
    with tempfile.TemporaryDirectory() as tmp:
        destino = Path(tmp) / PASTA_DE_TESTE
        destino.mkdir()
        if exe.is_dir():
            shutil.copytree(exe, destino / exe.name)
            executavel = destino / exe.name / f"{exe.name}.exe"
        else:
            shutil.copy2(exe, destino / exe.name)
            executavel = destino / exe.name
        captura = destino / "captura.png"
        ambiente = {**os.environ, "SDL_VIDEODRIVER": "dummy"}
        r = subprocess.run([str(executavel), "--piloto", "--quadros", str(QUADROS), "--captura", str(captura)],
                           capture_output=True, text=True, errors="replace", timeout=300, env=ambiente, cwd=destino)
        if r.returncode != 0:
            problemas.append(f"codigo de saida {r.returncode}: {r.stderr.strip()[-300:]}")
        if f"{QUADROS} quadros" not in r.stdout:
            problemas.append(f"linha de resumo ausente: {r.stdout.strip()[-200:]!r}")
        if not captura.exists() or captura.stat().st_size < 2000:
            problemas.append("captura de tela ausente ou vazia")
    print(f"   fumaca [{rotulo}]: {'OK' if not problemas else 'FALHOU'}")
    for p in problemas:
        print("   -", p)
    return problemas


def tamanho_mb(caminho):
    if caminho.is_file():
        return caminho.stat().st_size / 1e6
    return sum(f.stat().st_size for f in caminho.rglob("*") if f.is_file()) / 1e6


def main(argv=None):
    ap = argparse.ArgumentParser(description="Empacota o T-Rex Game 3D")
    grupo = ap.add_mutually_exclusive_group()
    grupo.add_argument("--so-pasta", action="store_true")
    grupo.add_argument("--so-unico", action="store_true")
    args = ap.parse_args(argv)

    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        print("PyInstaller nao encontrado: rode  pip install -r requirements-dev.txt")
        return 1

    entrega = SAIDA / "entrega"
    entrega.mkdir(parents=True, exist_ok=True)
    problemas = []

    if not args.so_unico:
        pyinstaller("--onedir", NOME_PASTA)
        pasta = SAIDA / "dist" / NOME_PASTA
        problemas += fumaca(pasta, "pasta")
        zip_pasta = shutil.make_archive(str(entrega / f"{NOME_PASTA}_pasta"), "zip", SAIDA / "dist", NOME_PASTA)
        print(f"   pasta {tamanho_mb(pasta):.0f} MB; zip {tamanho_mb(Path(zip_pasta)):.0f} MB: {zip_pasta}")
    if not args.so_pasta:
        pyinstaller("--onefile", NOME_UNICO)
        unico = SAIDA / "dist" / f"{NOME_UNICO}.exe"
        problemas += fumaca(unico, "arquivo unico")
        shutil.copy2(unico, entrega / unico.name)
        print(f"   arquivo unico {tamanho_mb(unico):.0f} MB: {entrega / unico.name}")

    print("\nTUDO OK" if not problemas else f"\n{len(problemas)} PROBLEMA(S): veja acima")
    return 1 if problemas else 0


if __name__ == "__main__":
    sys.exit(main())
