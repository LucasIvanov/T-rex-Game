"""Testes do ponto de entrada (main.py), sem janela (SDL_VIDEODRIVER=dummy)."""
import pytest

import main


@pytest.fixture(autouse=True)
def sem_janela(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")


def test_roda_quadros_com_piloto_e_salva_a_captura(tmp_path, capsys):
    captura = tmp_path / "quadro.png"
    assert main.main(["--piloto", "--quadros", "120", "--modo", "constante", "--captura", str(captura)]) == 0
    saida = capsys.readouterr().out
    assert "120 quadros" in saida and "estado=jogando" in saida and "modo=constante" in saida
    assert captura.stat().st_size > 2000


@pytest.mark.parametrize("modo", ["constante", "gouraud", "phong"])
def test_tres_modos_de_sombreamento(modo, capsys):
    assert main.main(["--piloto", "--quadros", "30", "--modo", modo]) == 0
    assert f"modo={modo}" in capsys.readouterr().out


def test_sem_piloto_fica_esperando(capsys):
    assert main.main(["--quadros", "30"]) == 0
    saida = capsys.readouterr().out
    assert "estado=esperando" in saida and "pontos=0" in saida


def test_fator_do_sol_acelera_so_o_sol(capsys):
    # 60 quadros de 1/60 s = 1 s: Sol em x = 6 - 0.2 * 1 = 5.80; com fator 10: 6 - 0.2 * 10 = 4.00
    assert main.main(["--quadros", "60"]) == 0
    assert "sol_x=5.80" in capsys.readouterr().out
    assert main.main(["--quadros", "60", "--fator-sol", "10"]) == 0
    saida = capsys.readouterr().out
    assert "sol_x=4.00" in saida and "tempo_de_jogo=0.0" in saida      # o relogio do jogo nao acelera


def test_fator_do_sol_invalido_e_rejeitado(capsys):
    with pytest.raises(SystemExit) as erro:
        main.main(["--fator-sol", "0.5"])
    assert erro.value.code == 2
    assert "deve ser >= 1" in capsys.readouterr().err


def test_modo_invalido_e_rejeitado():
    with pytest.raises(SystemExit) as erro:
        main.main(["--modo", "flat"])
    assert erro.value.code == 2


def test_piloto_joga_ate_a_primeira_esfera_passar(capsys):
    # a primeira esfera nasce em 1.0 s e passa pelo cubo em torno de 3.5 s: com 5 s o piloto ja fez 1 ponto
    assert main.main(["--piloto", "--quadros", "300", "--modo", "constante"]) == 0
    saida = capsys.readouterr().out
    assert "estado=jogando" in saida and "pontos=1" in saida
