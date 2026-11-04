# Roteiro da demonstração (cerca de 10 minutos)

Apresentação: 03/12/2026, Laboratório 1, 13h30 às 17h00. O enunciado (seção 8d) pede que a demonstração
evidencie **os movimentos dos objetos, o salto do personagem, a iluminação e o deslocamento do Sol**. Os
tempos abaixo são sugestões.

## Antes de ir (na véspera)

- [ ] Levar os arquivos de `_saida/entrega/` (`TRexGame3D_pasta.zip` e `TRexGame3D_unico.exe`) e o código, em
      pendrive e no repositório.
- [ ] Testar o zip e o `.exe` em outro computador (de preferência sem Python e com antivírus diferente).
- [ ] Conferir `python -m trex3d.aceite --estrito`: todos os casos em PASS, nenhum pendente.
- [ ] Conferir `python -m pytest`: tudo verde.
- [ ] Ensaiar uma vez com cronômetro. Decidir quem fala em cada parte.

## Sequência

| Tempo | O que fazer | O que dizer (e o que a banca vê) |
|---|---|---|
| 0:00 | Descompactar e abrir `TRexGame3D.exe` | Objetivo do trabalho; o console aberto é para mostrar erros. |
| 0:30 | **Tela de espera**, sem apertar nada | Cenário: terreno, céu, nuvens fixas (polígonos por scanline) e cactos de fundo em **cinco escalas** (extra de escala). O **Sol** é uma esfera amarela. |
| 1:30 | Apertar **espaço**: a partida começa | Esferas entram pela direita e **rolam** (rotação em z, ângulo = distância/raio) e **quicam**; tudo é função do tempo (independe do FPS). |
| 2:30 | Saltar por cima de 2 ou 3 esferas | **Salto** parabólico (subida e descida, volta ao chão). O placar sobe a cada esfera ultrapassada; mostrar o "HI". |
| 3:30 | Deixar bater de propósito | **Colisão** (ponto do cubo mais próximo do centro da esfera); aparece "Fim de jogo". |
| 4:00 | Apertar **espaço** de novo (depois de 0,5 s) | Reinício: pontos zerados, esferas novas; o recorde fica. |
| 4:30 | Segurar **T** por uns 10 s | **Deslocamento do Sol** (acelerado só para a demonstração; o normal é 30 vezes mais lento que as esferas). A **iluminação muda**: cor do terreno, das faces do cubo e do brilho nas esferas; o Sol passa atrás das nuvens e dos cactos. |
| 6:00 | Teclas **1**, **2**, **3** com uma esfera em primeiro plano | **Constante**: facetas. **Gouraud**: suave, mas sem o brilho (+10). **Phong**: brilho dentro da face (+20). O FPS aparece no canto. |
| 7:30 | Mostrar o relatório (fig. 1 e 2) e a estrutura do código | Pipeline: `R·T(−VRP)`, projeção ortográfica, janela→viewport com `y` invertido; faces de costas por `N·n > 0`; ordem do pintor; scanline com interpolação; fórmula `I = ka·Ia + fatt·Il·(kd·N·L + ks·(R·S)ⁿ)`. |
| 8:30 | No terminal: `python -m pytest` e `python -m trex3d.aceite --estrito` | 275 testes com resultados calculados à mão; casos de aceite conferidos contra as contas do grupo. |
| 9:30 | Perguntas | Ver a lista abaixo. |

Comandos úteis:

```bash
python main.py
```

```bash
python main.py --piloto --modo phong
```

```bash
python main.py --fator-sol 20
```

(`--piloto`: o jogo joga sozinho, útil se faltar tempo ou para falar enquanto roda. `--fator-sol 20`: o Sol
anda 20 vezes mais rápido durante toda a execução.)

## Se algo der errado

- O `.exe` não abre ou o antivírus bloqueia: rodar pelo código (`pip install -r requirements.txt` e
  `python main.py`). Deixar o ambiente virtual pronto antes.
- A máquina é lenta: usar o modo 1 ou 2 (mais leves) e dizer que o Phong foi medido a 49 FPS na máquina de desenvolvimento.
- Sem tempo para as 3 teclas: mostrar a figura 1 do relatório.

## Perguntas prováveis

- **Por que 20 graus de elevação e não uma vista exatamente frontal?** Numa vista frontal pura o cubo seria um
  quadrado; com a elevação aparecem frente e topo, e a ordem de pintura e a profundidade ficam visíveis. O eixo x
  continua horizontal na tela e o movimento é lateral. _[Resposta combinada com o professor.]_
- **Como a animação não depende do FPS?** Posições e ângulos são funções do tempo (parábola, `x = x0 − v·t`) e
  cada esfera nasce no instante exato; testamos 30 e 120 quadros por segundo e passos irregulares.
- **Como funciona a colisão se a apostila não a cobre?** Distância do centro da esfera ao ponto mais próximo do
  cubo menor que o raio; testada a cada 1/120 s para não atravessar. Casos à mão: esfera a 1,2 do centro apenas
  encosta; na quina, `dx = dy = 0,5` dá 0,707 > 0,6 e não colide.
- **Por que o terreno é sempre sombreado de forma constante?** É a maior área da tela; o Phong ali custava
  cerca de 3 vezes mais por quadro e não muda o resultado de um plano.
- **O que é diferente entre Gouraud e Phong?** O Gouraud interpola a cor calculada nos vértices; o Phong
  interpola a normal e calcula a luz em cada pixel, por isso reproduz o brilho dentro da face.
- **O que foi feito com IA?** Responder conforme a seção 1 do relatório (autorização do professor, em _[data]_).
