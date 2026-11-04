# T-Rex Game 3D

Trabalho 2 de Computação Gráfica (UNIOESTE, Ciência da Computação, 2026, prof. Adair Santa Catarina).

Integrantes: _(preencher com os nomes completos)_

Versão 3D do jogo "T-Rex Game" do Chrome: um **cubo** (o T-Rex) salta sobre **esferas** (os cactos) que rolam
e quicam da direita para a esquerda, com um **Sol** que cruza o céu e é a fonte de luz da cena. Projeção
ortográfica em vista frontal (pipeline de Alvy Ray Smith), iluminação ambiente + difusa + especular e três
sombreamentos (constante, Gouraud e Phong).

## Requisitos

- Python 3.10 ou superior (desenvolvido e testado com Python 3.13 no Windows 11; nas versões anteriores não foi testado).
- Bibliotecas: `pygame` 2.6 ou superior e `numpy` 2.0 ou superior (ver `requirements.txt`).

Todos os algoritmos de Computação Gráfica (transformações, câmera, ocultação de faces, preenchimento por
scanline, iluminação, sombreamento, colisão) estão no pacote `trex3d/` e não usam nenhuma biblioteca gráfica.
O `pygame` é usado só em `main.py`, para abrir a janela, ler o teclado, exibir o quadro já pronto e escrever
o texto; o `numpy` guarda vetores, matrizes e o framebuffer.

## Executar a partir do código

```bash
python -m venv .venv
```

Ative o ambiente (Windows: `.venv\Scripts\activate`; Linux e macOS: `source .venv/bin/activate`) e instale:

```bash
pip install -r requirements.txt
```

```bash
python main.py
```

## Executar o executável (Windows)

Duas formas, geradas por `python empacotamento/construir.py` (ver mais abaixo):

- **Pasta** `TRexGame3D_pasta.zip`: descompacte e abra `TRexGame3D\TRexGame3D.exe`. Abre rápido. Mantenha os
  arquivos da pasta juntos.
- **Arquivo único** `TRexGame3D_unico.exe`: um arquivo só; leva alguns segundos a mais para abrir, porque se
  extrai para a pasta temporária a cada execução.

Não é preciso instalar Python. Uma janela de console abre junto com o jogo de propósito: se algo falhar, a
mensagem de erro aparece nela. Como o executável não é assinado, o Windows ou o antivírus pode avisar; nesse
caso use a execução a partir do código (seção anterior).

## Controles

| Tecla | Ação |
|---|---|
| Barra de espaço | Inicia a partida, pula e (depois do fim de jogo) reinicia |
| 1, 2, 3 | Sombreamento constante, Gouraud, Phong (aparece no canto inferior esquerdo, com o FPS) |
| T (segurar) | Acelera o Sol 20 vezes, para a demonstração (só o Sol; o jogo segue no ritmo normal) |
| Esc | Sai |

O jogo termina quando o cubo bate em uma esfera. A pontuação conta as esferas ultrapassadas; o recorde fica
no canto superior direito ("HI").

## Opções da linha de comando

```
python main.py [--modo constante|gouraud|phong] [--piloto] [--fator-sol F] [--quadros N] [--captura arquivo.png]
```

- `--piloto`: o próprio jogo aperta a barra de espaço (serve para a demonstração).
- `--fator-sol F`: o Sol anda F vezes mais rápido (F de 1 em diante; padrão 1).
- `--quadros N`: roda N quadros com passo fixo de 1/60 s e sai, imprimindo um resumo. Para rodar sem janela,
  defina antes `SDL_VIDEODRIVER=dummy`.
- `--captura arquivo.png`: salva a última imagem.

## Testes e aceite

```bash
pip install -r requirements-dev.txt
python -m pytest
```

Os resultados esperados dos testes foram calculados à mão (as contas estão nos comentários de cada arquivo).
O conferidor dos casos de aceite do grupo:

```bash
python -m trex3d.aceite
python -m trex3d.aceite --estrito
```

Ele lê `tests/aceite/casos.json`; com `--estrito`, casos ainda sem resultado esperado contam como falha.

## Gerar os executáveis

```bash
pip install -r requirements-dev.txt
python empacotamento/construir.py
```

Gera tudo em `_saida/` (a pasta, o arquivo único e o zip em `_saida/entrega/`) e testa cada executável sem
janela, a partir de uma pasta com espaço e acento no nome. Use `--so-pasta` ou `--so-unico` para gerar um só.

## Estrutura

| Arquivo | Conteúdo |
|---|---|
| `main.py` | Janela, teclado, texto na tela e laço do jogo |
| `trex3d/mat.py` | Vetores e matrizes 4x4 homogêneas (translação, escala, rotação, composição) |
| `trex3d/camera.py` | Câmera ortográfica (sistema do observador) e janela→viewport |
| `trex3d/modelos.py` | Malhas poliédricas: cubo e caixa, esfera por faixas e fatias, plano, cacto |
| `trex3d/visibilidade.py` | Faces de costas pela normal e ordem do pintor |
| `trex3d/raster.py` | Preenchimento de polígonos por scanline com interpolação |
| `trex3d/luz.py` | Iluminação ambiente, difusa e especular; materiais |
| `trex3d/sombreamento.py` | Constante, Gouraud e Phong |
| `trex3d/mundo.py` | Animação por tempo: salto, esferas, Sol |
| `trex3d/jogo.py` | Estados, colisão, pontuação |
| `trex3d/cenario.py` | Nuvens fixas e cactos de fundo em escalas diferentes |
| `trex3d/cena.py` | Montagem do quadro |
| `trex3d/aceite.py` | Conferidor dos casos calculados à mão |
| `tests/` | Testes automáticos e casos de aceite |
| `relatorio/` | Relatório técnico, roteiro da demonstração e figuras |
| `empacotamento/construir.py` | Geração e teste dos executáveis |
