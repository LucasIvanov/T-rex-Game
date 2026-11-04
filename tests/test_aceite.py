"""Testes do runner de aceite (passo 17 do plano).

O runner confere casos calculados A MAO pelo usuario contra o codigo. Aqui se testa o runner, com entradas
e resultados esperados DIFERENTES dos sugeridos em tests/aceite/casos.json (para nao revelar as respostas).
Contas a mao dos casos de teste:
  projecao: escala 2 e translacao (1,0,0) em (1,1,1) -> (3,2,2); camera frontal, 100 px por unidade:
            px = 400 + 300 = 700, py = 600 - (2 + 3)*100 = 100, profundidade 2.
            rot_z 90 graus em (1,0,0) -> (0,1,0): px = 400, py = 600 - 400 = 200, profundidade 0.
  iluminacao: normal (0,0,2) vira (0,0,1); Sol em (0,0,10): N.L = 1; ka = 0.1, kd = 0.5, ks = 0 -> 0.6 em cada canal.
  rotacao em torno de ponto: eixo x, 90 graus, centro (0,1,1), ponto (0,2,1): relativo (0,1,0) -> (0,0,1) -> (0,1,2).
  colisao: cubo (0,0.6,0) lado 1.2, esfera r = 0.6 em (1.1,0.6,0) colide; em (1.3,0.6,0) nao.
  preenchimento: triangulo (0,0),(4,0),(0,4) -> 10 pixels.
"""
import json
import pathlib

import pytest

from trex3d import aceite

CAMERA = {"vrp": [0, 0, 0], "n": [0, 0, 1], "view_up": [0, 1, 0],
          "janela": [-4, -3, 4, 3], "viewport": [0, 0, 800, 600]}
MATERIAL = {"ka": [0.1, 0.1, 0.1], "kd": [0.5, 0.5, 0.5], "ks": [0, 0, 0], "n": 1}


def caso(id, tipo, entrada, esperado, **extra):
    return {"id": id, "tipo": tipo, "descricao": f"caso {id}", "entrada": entrada, "esperado": esperado, **extra}


PROJ_ESCALA = caso("p1", "projecao", {"camera": CAMERA, "ponto": [1, 1, 1],
                                      "transformacoes": [["escala", 2, 2, 2], ["translacao", 1, 0, 0]]}, [700, 100, 2])
PROJ_ROT = caso("p2", "projecao", {"camera": CAMERA, "ponto": [1, 0, 0], "transformacoes": [["rot_z", 90]]}, [400, 200, 0])
ILUM = caso("i1", "iluminacao", {"material": MATERIAL, "luz": {"posicao": [0, 0, 10]}, "ponto": [0, 0, 0],
                                 "normal": [0, 0, 2], "observador": [0, 0, 1]}, [0.6, 0.6, 0.6])
ROT = caso("r1", "rotacao_em_torno_de_ponto", {"eixo": "x", "graus": 90, "centro": [0, 1, 1], "ponto": [0, 2, 1]}, [0, 1, 2])
COL_SIM = caso("c1", "colisao", {"centro_cubo": [0, 0.6, 0], "lado": 1.2, "centro_esfera": [1.1, 0.6, 0], "raio": 0.6}, True)
COL_NAO = caso("c2", "colisao", {"centro_cubo": [0, 0.6, 0], "lado": 1.2, "centro_esfera": [1.3, 0.6, 0], "raio": 0.6}, False)
PREENCH = caso("f1", "preenchimento", {"vertices": [[0, 0], [4, 0], [0, 4]]}, 10)
TODOS = [PROJ_ESCALA, PROJ_ROT, ILUM, ROT, COL_SIM, COL_NAO, PREENCH]


def avaliar(c):
    return aceite.avaliar_caso(c)


@pytest.mark.parametrize("c", TODOS, ids=[c["id"] for c in TODOS])
def test_cada_tipo_confere_com_a_conta_a_mao(c):
    assert avaliar(c).situacao == "PASS"


def test_resultado_errado_e_fail():
    errado = caso("x", "projecao", PROJ_ESCALA["entrada"], [700, 101, 2])
    r = avaliar(errado)
    assert r.situacao == "FAIL" and r.obtido == pytest.approx([700, 100, 2])


def test_colisao_errada_e_fail():
    assert avaliar(caso("x", "colisao", COL_SIM["entrada"], False)).situacao == "FAIL"


def test_tolerancia_padrao_e_por_caso():
    perto = caso("x", "projecao", PROJ_ESCALA["entrada"], [700.3, 100, 2])
    assert avaliar(perto).situacao == "FAIL"                                  # padrao 0.01
    assert avaliar({**perto, "tolerancia": 0.5}).situacao == "PASS"


def test_sem_esperado_e_pendente_e_nao_revela_o_obtido():
    r = avaliar(caso("x", "projecao", PROJ_ESCALA["entrada"], None))
    assert r.situacao == "PENDENTE" and r.obtido is None


def test_pendente_com_entrada_invalida_vira_erro():
    r = avaliar(caso("x", "projecao", {"camera": CAMERA}, None))               # falta "ponto"
    assert r.situacao == "ERRO"


def test_tipo_desconhecido_e_erro():
    assert avaliar(caso("x", "levitacao", {}, 1)).situacao == "ERRO"


def test_entrada_incompleta_e_erro_com_mensagem():
    r = avaliar(caso("x", "colisao", {"lado": 1.2}, True))
    assert r.situacao == "ERRO" and "centro_cubo" in r.detalhe


def escrever(tmp_path, casos):
    arq = tmp_path / "casos.json"
    arq.write_text(json.dumps({"casos": casos}), encoding="utf-8")
    return str(arq)


def test_saida_todos_pass_codigo_0(tmp_path, capsys):
    assert aceite.main([escrever(tmp_path, TODOS)]) == 0
    saida = capsys.readouterr().out
    linhas = saida.splitlines()
    assert sum(l.startswith("PASS") for l in linhas) == 7 and not any(l.startswith("FAIL") for l in linhas)
    assert "Total: 7 | PASS: 7 | FAIL: 0 | PENDENTE: 0" in saida


def test_saida_com_fail_codigo_1(tmp_path, capsys):
    errado = caso("ruim", "colisao", COL_SIM["entrada"], False)
    assert aceite.main([escrever(tmp_path, [COL_SIM, errado])]) == 1
    saida = capsys.readouterr().out
    assert "FAIL" in saida and "ruim" in saida and "esperado" in saida and "obtido" in saida
    assert "PASS: 1 | FAIL: 1" in saida


def test_pendentes_nao_falham_a_menos_que_estrito(tmp_path, capsys):
    pend = caso("falta", "colisao", COL_SIM["entrada"], None)
    arq = escrever(tmp_path, [COL_SIM, pend])
    assert aceite.main([arq]) == 0
    assert "PENDENTE: 1" in capsys.readouterr().out
    assert aceite.main([arq, "--estrito"]) == 2


def test_pendente_nao_imprime_o_valor_calculado(tmp_path, capsys):
    pend = caso("falta", "projecao", PROJ_ESCALA["entrada"], None)       # o codigo calcularia (700, 100, 2)
    aceite.main([escrever(tmp_path, [pend])])
    saida = capsys.readouterr().out
    assert "PENDENTE" in saida and "falta" in saida
    assert "700" not in saida and "obtido" not in saida


def test_erro_conta_como_falha_codigo_1(tmp_path):
    assert aceite.main([escrever(tmp_path, [caso("x", "levitacao", {}, 1)])]) == 1


def test_arquivo_inexistente_e_json_invalido(tmp_path):
    assert aceite.main([str(tmp_path / "nao_existe.json")]) == 1
    ruim = tmp_path / "ruim.json"
    ruim.write_text("{ nao e json", encoding="utf-8")
    assert aceite.main([str(ruim)]) == 1


def test_ids_repetidos_sao_erro(tmp_path):
    assert aceite.main([escrever(tmp_path, [COL_SIM, COL_SIM])]) == 1


ARQUIVO_DO_PROJETO = pathlib.Path(aceite.__file__).resolve().parents[1] / "tests" / "aceite" / "casos.json"


def test_arquivo_do_projeto_existe_e_tem_casos_validos():
    dados = json.loads(ARQUIVO_DO_PROJETO.read_text(encoding="utf-8"))
    casos = dados["casos"]
    assert 5 <= len(casos) <= 12
    assert len({c["id"] for c in casos}) == len(casos)
    tipos = {c["tipo"] for c in casos}
    assert {"projecao", "iluminacao", "rotacao_em_torno_de_ponto", "colisao"} <= tipos
    for c in casos:
        r = avaliar(c)
        assert r.situacao in ("PASS", "FAIL", "PENDENTE"), (c["id"], r.detalhe)      # entrada valida, sem ERRO
