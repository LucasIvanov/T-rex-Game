"""Testes dos sombreamentos (parte 7 do plano): constante, Gouraud e Phong.

Caso central, a mao. Triangulo na tela (0,0),(8,0),(0,8); vertices do mundo todos na origem; normais dos
vertices: v0 = (0,0,1), v1 = (1,0,0). Sol em 10*(c,0,c) e observador S = (c,0,c), c = raiz(2)/2 (a bissetriz
das duas normais). Material M: ka=0.2, kd=(0.5,0.4,0.3), ks=0.3, n=2.
  - Em v0 e em v1: N.L = c; R = (-c,0,c) (resp. (c,0,-c)); R.S = 0 -> I0 = I1 = ka + kd*c (sem brilho).
  - Gouraud no meio da aresta (pixel (4,0)): media de I0 e I1 = I0.
  - Phong no mesmo pixel: normal interpolada (0.5,0,0.5), normalizada (c,0,c): N.L = 1, R = N, R.S = 1
    -> I = ka + kd + ks = (1.0, 0.9, 0.8): o brilho aparece DENTRO da face, onde Gouraud e constante nao o veem.
Conferir as contas e acrescentar casos proprios em CASOS_DO_USUARIO.
"""
import math

import numpy as np
import pytest

from trex3d import camera, luz, mat, modelos, sombreamento

C = math.sqrt(2) / 2
M = luz.Material(ka=(0.2, 0.2, 0.2), kd=(0.5, 0.4, 0.3), ks=(0.3, 0.3, 0.3), n=2)
VERDE = luz.Material(ka=(0, 0, 0), kd=(0, 1, 0), ks=(0, 0, 0), n=1)
SOL = luz.Luz(posicao=(10 * C, 0, 10 * C))
S = (C, 0, C)
I0 = np.array([0.2 + 0.5 * C, 0.2 + 0.4 * C, 0.2 + 0.3 * C])   # cor em v0 e v1 (sem brilho)
MAXIMO = np.array([1.0, 0.9, 0.8])                              # ka + kd + ks


def aprox(v, tol=1e-9):
    return pytest.approx(v, abs=tol)


def triangulo(normais, fn=(0, 0, 1), ids=0, P=None):
    P = [(0, 0), (8, 0), (0, 8), (0, 8)] if P is None else P
    return sombreamento.Faces(
        P=np.array([P], float),
        PV=np.zeros((1, 4, 3)),
        NV=np.array([[normais[0], normais[1], normais[2], normais[2]]], float),
        FN=np.array([fn], float),
        ids=np.array([ids]),
    )


def cor_do_pixel(modo, faces, x, y, materiais=(M,), luz_=SOL, s=S, w=20, h=20):
    px, py, poly, rgb = sombreamento.calcular_cores(modo, faces, list(materiais), luz_, s, w, h)
    achado = (px == x) & (py == y)
    assert achado.sum() == 1
    return rgb[achado][0]


TRI = triangulo([(0, 0, 1), (1, 0, 0), (0, 1, 0)])


def test_constante_uma_cor_so_calculada_no_centro_com_a_normal_da_face():
    # centro na origem, FN = z: N.L = c, igual a I0; todos os 36 pixels (8+7+...+1) com a mesma cor
    px, py, poly, rgb = sombreamento.calcular_cores("constante", TRI, [M], SOL, S, 20, 20)
    assert px.size == 36
    assert rgb == aprox(np.tile(I0, (36, 1)))


def test_constante_ignora_as_normais_dos_vertices():
    outra = triangulo([(1, 0, 0), (0, 1, 0), (0, 0, -1)])
    assert cor_do_pixel("constante", outra, 1, 1) == aprox(I0)


def test_gouraud_no_vertice_tem_a_cor_do_vertice():
    assert cor_do_pixel("gouraud", TRI, 0, 0) == aprox(I0)


def test_gouraud_meio_da_aresta_e_a_media_das_pontas():
    # I0 e I1 iguais (sem brilho) -> pixel (4,0) = I0; sem brilho, ao contrario do Phong
    assert cor_do_pixel("gouraud", TRI, 4, 0) == aprox(I0)


def test_gouraud_media_de_pontas_diferentes():
    # v0: N = z, luz de frente (Sol em +z): (1.0, 0.9, 0.8); v1: N = -z: so ambiente 0.2
    # pixel (4,0): u = 0.5 -> (0.6, 0.55, 0.5)
    f = triangulo([(0, 0, 1), (0, 0, -1), (1, 0, 0)])
    sol_z = luz.Luz(posicao=(0, 0, 10))
    assert cor_do_pixel("gouraud", f, 0, 0, luz_=sol_z, s=(0, 0, 1)) == aprox(MAXIMO)
    assert cor_do_pixel("gouraud", f, 4, 0, luz_=sol_z, s=(0, 0, 1)) == aprox([0.6, 0.55, 0.5])


def test_phong_no_vertice_e_igual_ao_gouraud():
    assert cor_do_pixel("phong", TRI, 0, 0) == aprox(cor_do_pixel("gouraud", TRI, 0, 0))


def test_phong_mostra_o_brilho_dentro_da_face():
    assert cor_do_pixel("phong", TRI, 4, 0) == aprox(MAXIMO)
    assert cor_do_pixel("gouraud", TRI, 4, 0) == aprox(I0)
    assert cor_do_pixel("constante", TRI, 4, 0) == aprox(I0)


def test_phong_normaliza_a_normal_interpolada():
    # sem normalizar, (0.5,0,0.5) daria N.L = c e nao 1: a cor maxima prova a normalizacao
    assert cor_do_pixel("phong", TRI, 4, 0)[0] == aprox(1.0)


def test_materiais_diferentes_por_face():
    # duas faces separadas (ids 0 e 1), normal z, Sol em +z: M -> (1.0, 0.9, 0.8); verde -> (0, 1, 0)
    sol_z = luz.Luz(posicao=(0, 0, 10))
    f1 = triangulo([(0, 0, 1)] * 3, ids=0)
    f2 = triangulo([(0, 0, 1)] * 3, ids=1, P=[(10, 0), (18, 0), (10, 8), (10, 8)])
    juntas = sombreamento.juntar([f1, f2])
    for modo in sombreamento.MODOS:
        assert cor_do_pixel(modo, juntas, 1, 1, materiais=(M, VERDE), luz_=sol_z, s=(0, 0, 1)) == aprox(MAXIMO)
        assert cor_do_pixel(modo, juntas, 11, 1, materiais=(M, VERDE), luz_=sol_z, s=(0, 0, 1)) == aprox([0, 1, 0])


def test_centro_de_triangulo_usa_so_os_3_vertices_distintos():
    # (0,0,0),(3,0,0),(0,3,0) com o ultimo repetido: media de 3 = (1,1,0); a de 4 seria (0.75,0.75,0)
    pv = np.array([[(0, 0, 0), (3, 0, 0), (0, 3, 0), (0, 3, 0)], [(0, 0, 0), (2, 0, 0), (2, 2, 0), (0, 2, 0)]], float)
    assert sombreamento._centros(pv) == aprox(np.array([[1, 1, 0], [1, 1, 0]], float))


def test_modo_desconhecido_e_erro():
    with pytest.raises(ValueError):
        sombreamento.calcular_cores("flat", TRI, [M], SOL, S, 20, 20)


def quadrado(p0, p1, ids):
    (x0, y0), (x1, y1) = p0, p1
    return sombreamento.Faces(
        P=np.array([[(x0, y0), (x1, y0), (x1, y1), (x0, y1)]], float),
        PV=np.array([[(-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)]], float),
        NV=np.tile([0.0, 0.0, 1.0], (1, 4, 1)),
        FN=np.array([[0, 0, 1]], float),
        ids=np.array([ids]),
    )


def test_pintar_faces_ordem_do_pintor_a_ultima_cobre():
    sol_z = luz.Luz(posicao=(0, 0, 10))
    fb = np.zeros((10, 10, 3), np.uint8)
    faces = sombreamento.juntar([quadrado((0, 0), (6, 6), 0), quadrado((3, 3), (9, 9), 1)])
    sombreamento.pintar_faces(fb, "constante", faces, [M, VERDE], sol_z, (0, 0, 1))
    # so a primeira; 0.9 * 255 = 229.5 e um empate de arredondamento, entao aceita-se 1 nivel de diferenca
    assert np.abs(fb[1, 1].astype(int) - luz.para_uint8(MAXIMO).astype(int)).max() <= 1
    assert tuple(fb[4, 4]) == (0, 255, 0)                       # sobreposicao: vence a segunda
    assert tuple(fb[8, 1]) == (0, 0, 0)


def test_pintar_faces_devolve_o_numero_de_pixels():
    fb = np.zeros((10, 10, 3), np.uint8)
    n = sombreamento.pintar_faces(fb, "gouraud", quadrado((2, 2), (6, 6), 0), [M], SOL, S)
    assert n == 16


def test_juntar_lista_vazia():
    vazio = sombreamento.juntar([])
    assert vazio.P.shape == (0, 4, 2)
    fb = np.zeros((10, 10, 3), np.uint8)
    assert sombreamento.pintar_faces(fb, "phong", vazio, [M], SOL, S) == 0


JANELA, VIEWPORT = (-4, -3, 4, 3), (0, 0, 800, 600)


def cam(n):
    return camera.CameraOrtografica((0, 0, 0), n, (0, 1, 0), JANELA, VIEWPORT)


def test_preparar_faces_cubo_a_30_graus():
    f = sombreamento.preparar_faces(modelos.criar_cubo(2), np.eye(4), cam((0, 0.5, math.sqrt(3) / 2)), 3)
    assert f.P.shape == (2, 4, 2) and f.PV.shape == (2, 4, 3) and f.ids.tolist() == [3, 3]
    normais = {tuple(int(round(x)) for x in n) for n in f.FN}
    assert normais == {(0, 0, 1), (0, 1, 0)}
    assert f.NV[0] == aprox(np.tile(f.FN[0], (4, 1)))   # cubo: normal do vertice = normal da face


def test_preparar_faces_projeta_com_a_camera():
    # vista frontal, cubo de lado 2 na origem: face +z vai de (-1,-1) a (1,1) -> px 300..500, py 200..400
    f = sombreamento.preparar_faces(modelos.criar_cubo(2), np.eye(4), cam((0, 0, 1)), 0)
    assert f.P.shape[0] == 1
    assert {tuple(np.round(p, 6)) for p in f.P[0]} == {(300.0, 200.0), (300.0, 400.0), (500.0, 200.0), (500.0, 400.0)}


def test_preparar_faces_gira_as_normais():
    # Ry(90): +z -> +x. Com o observador em +x so a face que era +z fica visivel, agora com normal +x
    f = sombreamento.preparar_faces(modelos.criar_cubo(2), mat.rot_y(math.pi / 2), cam((1, 0, 0)), 0)
    assert f.FN.shape[0] == 1
    assert f.FN[0] == aprox([1, 0, 0])
    assert f.NV[0] == aprox(np.tile([1.0, 0, 0], (4, 1)))


def test_preparar_faces_escala_uniforme_mantem_normais_unitarias():
    f = sombreamento.preparar_faces(modelos.criar_esfera(1, 6, 8), mat.escala(3, 3, 3), cam((0, 0, 1)), 0)
    assert np.linalg.norm(f.NV, axis=-1) == aprox(np.ones(f.NV.shape[:2]))
    assert np.abs(f.PV).max() == aprox(3.0)


def test_preparar_faces_escala_nao_uniforme_e_erro():
    with pytest.raises(ValueError):
        sombreamento.preparar_faces(modelos.criar_cubo(2), mat.escala(2, 1, 1), cam((0, 0, 1)), 0)


def esfera_pintada(modo):
    c = camera.CameraOrtografica((0, 0, 0), (0, 0, 1), (0, 1, 0), (-2, -1.5, 2, 1.5), (0, 0, 160, 120))
    faces = sombreamento.preparar_faces(modelos.criar_esfera(1, 6, 8), np.eye(4), c, 0)
    fb = np.zeros((160, 120, 3), np.uint8)
    sombreamento.pintar_faces(fb, modo, faces, [luz.MATERIAL_ESFERA], luz.Luz(posicao=(3, 4, 5)), c.n)
    return fb, faces


def test_esfera_mesma_silhueta_nos_tres_modos():
    mascaras = [esfera_pintada(m)[0].any(axis=-1) for m in sombreamento.MODOS]
    assert np.array_equal(mascaras[0], mascaras[1]) and np.array_equal(mascaras[0], mascaras[2])
    assert mascaras[0].sum() > 3000     # raio de 40 px: area de uns 4000 a 5000 pixels


def test_esfera_constante_tem_uma_cor_por_face_e_phong_e_liso():
    fb_c, faces = esfera_pintada("constante")
    fb_p, _ = esfera_pintada("phong")
    cores_c = {tuple(p) for p in fb_c.reshape(-1, 3) if p.any()}
    cores_p = {tuple(p) for p in fb_p.reshape(-1, 3) if p.any()}
    assert len(cores_c) <= faces.P.shape[0]      # no maximo uma cor por face visivel (24)
    assert len(cores_p) > 100                    # gradiente continuo


# Acrescente 2 a 3 casos seus: (modo, normais dos 3 vertices, pixel (x, y), (R, G, B) esperado), usando o
# triangulo (0,0),(8,0),(0,8), o Sol e o observador deste arquivo e o material M.
CASOS_DO_USUARIO = []


@pytest.mark.parametrize("modo, normais, pixel, esperado", CASOS_DO_USUARIO)
def test_casos_do_usuario(modo, normais, pixel, esperado):
    assert cor_do_pixel(modo, triangulo(normais), *pixel) == aprox(esperado)
