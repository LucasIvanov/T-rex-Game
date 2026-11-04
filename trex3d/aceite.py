"""Runner de aceite: confere o codigo com resultados calculados A MAO pelo usuario.

Uso: python -m trex3d.aceite [arquivo.json] [--estrito]
Padrao: tests/aceite/casos.json. Codigos de saida: 0 sem falhas; 1 com FAIL ou ERRO (ou arquivo invalido);
2 so com --estrito e ainda ha casos PENDENTES.

Formato: {"casos": [{"id", "tipo", "descricao", "entrada", "esperado", "tolerancia"?}, ...]}
Tipos e entradas:
  projecao: camera {vrp, n, view_up, janela [xmin,ymin,xmax,ymax], viewport [umin,vmin,umax,vmax]}, ponto [x,y,z],
            transformacoes [[op, ...], ...] aplicadas NA ORDEM DA LISTA (ops: translacao dx dy dz | escala sx sy sz |
            rot_x/rot_y/rot_z graus); esperado [px, py, profundidade] (profundidade = z do SRC, maior = mais perto).
  iluminacao: material {ka, kd, ks, n}, luz {posicao, ...}, ponto, normal, observador (normal e observador sao
            normalizados); esperado [R, G, B] em [0, 1] (valores acima de 1 aparecem como 1).
  rotacao_em_torno_de_ponto: eixo "x"|"y"|"z", graus, centro, ponto; esperado [x, y, z].
  colisao: centro_cubo, lado, centro_esfera, raio; esperado true/false.
  preenchimento: vertices [[x, y], ...], tela [largura, altura] (padrao [100, 100]); esperado = numero de pixels.
"esperado": null deixa o caso PENDENTE: o runner nao mostra o valor que o codigo calcularia.
"""
import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from trex3d import camera, jogo, luz, mat, raster

ARQUIVO_PADRAO = Path(__file__).resolve().parents[1] / "tests" / "aceite" / "casos.json"
TOLERANCIA_PADRAO = 0.01

OPERACOES = {
    "translacao": lambda dx, dy, dz: mat.translacao(dx, dy, dz),
    "escala": lambda sx, sy, sz: mat.escala(sx, sy, sz),
    "rot_x": lambda graus: mat.rot_x(math.radians(graus)),
    "rot_y": lambda graus: mat.rot_y(math.radians(graus)),
    "rot_z": lambda graus: mat.rot_z(math.radians(graus)),
}
ROTACOES = {"x": mat.rot_x, "y": mat.rot_y, "z": mat.rot_z}


def _matriz_do_modelo(operacoes):
    m = np.eye(4)
    for nome, *args in operacoes:
        if nome not in OPERACOES:
            raise ValueError(f"operacao desconhecida: {nome!r}")
        m = OPERACOES[nome](*args) @ m
    return m


def _projecao(e):
    c = e["camera"]
    cam = camera.CameraOrtografica(c["vrp"], c["n"], c["view_up"], c["janela"], c["viewport"])
    ponto = mat.aplicar(_matriz_do_modelo(e.get("transformacoes", [])), e["ponto"])
    return cam.projetar(ponto).tolist()


def _iluminacao(e):
    material = luz.Material(**e["material"])
    fonte = luz.Luz(**e["luz"])
    rgb = luz.iluminar(e["ponto"], mat.normalizar(e["normal"]), material, fonte, mat.normalizar(e["observador"]))
    return rgb.tolist()


def _rotacao_em_torno_de_ponto(e):
    rot = ROTACOES[e["eixo"]](math.radians(e["graus"]))
    return mat.aplicar(mat.em_torno_de(rot, e["centro"]), e["ponto"]).tolist()


def _colisao(e):
    return bool(jogo.colisao_cubo_esfera(e["centro_cubo"], e["lado"], e["centro_esfera"], e["raio"]))


def _preenchimento(e):
    largura, altura = e.get("tela", [100, 100])
    px = raster.rasterizar(np.array([e["vertices"]], dtype=float), None, largura, altura)[0]
    return int(px.size)


TIPOS = {
    "projecao": _projecao,
    "iluminacao": _iluminacao,
    "rotacao_em_torno_de_ponto": _rotacao_em_torno_de_ponto,
    "colisao": _colisao,
    "preenchimento": _preenchimento,
}


@dataclass
class Resultado:
    id: str
    descricao: str
    situacao: str            # PASS, FAIL, PENDENTE ou ERRO
    esperado: object = None
    obtido: object = None
    detalhe: str = ""


def _iguais(obtido, esperado, tolerancia):
    if isinstance(obtido, bool) or isinstance(esperado, bool):
        return isinstance(obtido, bool) and isinstance(esperado, bool) and obtido == esperado
    if isinstance(obtido, int):
        return obtido == esperado
    a, b = np.asarray(obtido, dtype=float), np.asarray(esperado, dtype=float)
    return a.shape == b.shape and bool(np.allclose(a, b, rtol=0.0, atol=tolerancia))


def avaliar_caso(caso):
    id_, descricao = caso.get("id", "?"), caso.get("descricao", "")
    try:
        executar = TIPOS.get(caso["tipo"])
        if executar is None:
            raise ValueError(f"tipo de caso desconhecido: {caso['tipo']!r}")
        obtido = executar(caso["entrada"])
    except KeyError as falta:
        return Resultado(id_, descricao, "ERRO", detalhe=f"falta o campo {falta}")
    except Exception as erro:
        return Resultado(id_, descricao, "ERRO", detalhe=f"{type(erro).__name__}: {erro}")

    esperado = caso.get("esperado")
    if esperado is None:
        return Resultado(id_, descricao, "PENDENTE")
    try:
        ok = _iguais(obtido, esperado, caso.get("tolerancia", TOLERANCIA_PADRAO))
    except (TypeError, ValueError) as erro:
        return Resultado(id_, descricao, "ERRO", esperado, detalhe=f"esperado em formato invalido: {erro}")
    return Resultado(id_, descricao, "PASS" if ok else "FAIL", esperado, obtido)


def _fmt(v):
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_fmt(x) for x in v) + "]"
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


def _imprimir(r, arquivo):
    print(f"{r.situacao:<9} {r.id:<10} {r.descricao}")
    if r.situacao in ("PASS", "FAIL"):
        print(f"          esperado: {_fmt(r.esperado)}   obtido: {_fmt(r.obtido)}")
    elif r.situacao == "PENDENTE":
        print(f'          falta o resultado: calcule a mao e preencha "esperado" em {arquivo}')
    else:
        print(f"          {r.detalhe}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m trex3d.aceite", description="Confere casos calculados a mao.")
    ap.add_argument("arquivo", nargs="?", default=str(ARQUIVO_PADRAO))
    ap.add_argument("--estrito", action="store_true", help="casos pendentes tambem contam como falha (codigo 2)")
    args = ap.parse_args(argv)

    try:
        casos = json.loads(Path(args.arquivo).read_text(encoding="utf-8"))["casos"]
    except (OSError, ValueError, KeyError, TypeError) as erro:
        print(f"ERRO ao ler {args.arquivo}: {erro}")
        return 1
    ids = [c.get("id") for c in casos]
    if len(set(ids)) != len(ids):
        print("ERRO: ha ids de casos repetidos")
        return 1

    resultados = [avaliar_caso(c) for c in casos]
    for r in resultados:
        _imprimir(r, args.arquivo)
    conta = {s: sum(r.situacao == s for r in resultados) for s in ("PASS", "FAIL", "PENDENTE", "ERRO")}
    resumo = f"Total: {len(resultados)} | PASS: {conta['PASS']} | FAIL: {conta['FAIL']} | PENDENTE: {conta['PENDENTE']}"
    if conta["ERRO"]:
        resumo += f" | ERRO: {conta['ERRO']}"
    print("\n" + resumo)
    if conta["FAIL"] or conta["ERRO"]:
        return 1
    if args.estrito and conta["PENDENTE"]:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
