"""Iluminacao (apostila Un. 8, secoes 8.1.1 a 8.1.5): ambiente + difusa (Lambert) + especular (Phong).

    I = ka*Ia + fatt*Il*(kd*max(0, N.L) + ks*max(0, R.S)^n) + emissao        (por canal R, G, B)
    L = (Sol - P)/|Sol - P|,  R = 2 (N.L) N - L,  fatt = min(1/(c1 + c2 d + c3 d^2), 1)

Normal N e direcao do observador S devem ser unitarias. Na projecao ortografica S e constante
(= camera.n). O brilho especular so existe se N.L > 0. O resultado fica cortado em [0, 1].
"""
from dataclasses import dataclass

import numpy as np


def _rgb(valor, nome):
    v = np.asarray(valor, dtype=float)
    if v.shape != (3,):
        raise ValueError(f"{nome} deve ter 3 componentes (R, G, B)")
    if np.any(v < 0):
        raise ValueError(f"{nome} nao pode ser negativo")
    return v


@dataclass
class Material:
    ka: object
    kd: object
    ks: object
    n: float
    emissao: object = (0.0, 0.0, 0.0)

    def __post_init__(self):
        self.ka = _rgb(self.ka, "ka")
        self.kd = _rgb(self.kd, "kd")
        self.ks = _rgb(self.ks, "ks")
        self.emissao = _rgb(self.emissao, "emissao")
        if self.n <= 0:
            raise ValueError("expoente especular n deve ser positivo")


@dataclass
class Luz:
    posicao: object
    intensidade: object = (1.0, 1.0, 1.0)
    ambiente: object = (1.0, 1.0, 1.0)
    c1: float = 1.0
    c2: float = 0.0
    c3: float = 0.0

    def __post_init__(self):
        self.posicao = np.asarray(self.posicao, dtype=float)
        self.intensidade = _rgb(self.intensidade, "intensidade")
        self.ambiente = _rgb(self.ambiente, "ambiente")
        if min(self.c1, self.c2, self.c3) < 0 or (self.c1 == 0 and self.c2 == 0 and self.c3 == 0):
            raise ValueError("constantes de atenuacao: nao negativas e nao todas nulas")


def iluminar(pontos, normais, material, luz, direcao_do_observador):
    """Intensidade RGB em [0, 1] para pontos (..., 3) com normais unitarias (..., 3)."""
    p = np.asarray(pontos, dtype=float)
    normal = np.asarray(normais, dtype=float)
    s = np.asarray(direcao_do_observador, dtype=float)

    ate_a_luz = luz.posicao - p
    d = np.linalg.norm(ate_a_luz, axis=-1, keepdims=True)
    if np.any(d == 0):
        raise ValueError("ponto na mesma posicao da fonte de luz")
    l = ate_a_luz / d
    nl = np.sum(normal * l, axis=-1, keepdims=True)
    difusa = np.maximum(nl, 0.0)
    r = 2.0 * nl * normal - l
    rs = np.maximum(np.sum(r * s, axis=-1, keepdims=True), 0.0)
    especular = np.where(nl > 0, rs ** material.n, 0.0)
    fatt = np.minimum(1.0 / (luz.c1 + luz.c2 * d + luz.c3 * d * d), 1.0)

    i = (material.ka * luz.ambiente
         + fatt * luz.intensidade * (material.kd * difusa + material.ks * especular)
         + material.emissao)
    return np.clip(i, 0.0, 1.0)


def para_uint8(intensidade):
    """Intensidade em [0, 1] -> inteiro 0..255 (arredonda para o mais proximo)."""
    return (np.clip(np.asarray(intensidade, dtype=float), 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def _fosco(cor, ka=0.25, kd=0.75):
    c = np.asarray(cor, dtype=float)
    return c * ka, c * kd


_ka, _kd = _fosco((0.75, 0.75, 0.78))
MATERIAL_CUBO = Material(ka=_ka, kd=_kd, ks=(0.2, 0.2, 0.2), n=10)
_ka, _kd = _fosco((0.30, 0.55, 0.20))
MATERIAL_ESFERA = Material(ka=_ka, kd=_kd, ks=(0.5, 0.5, 0.5), n=30)
_ka, _kd = _fosco((0.80, 0.55, 0.30))
MATERIAL_TERRENO = Material(ka=_ka, kd=_kd, ks=(0.0, 0.0, 0.0), n=1)
_ka, _kd = _fosco((0.18, 0.42, 0.22))
MATERIAL_CACTO = Material(ka=_ka, kd=_kd, ks=(0.1, 0.1, 0.1), n=5)
MATERIAL_SOL = Material(ka=(0, 0, 0), kd=(0, 0, 0), ks=(0, 0, 0), n=1, emissao=(1.0, 0.9, 0.2))
del _ka, _kd
