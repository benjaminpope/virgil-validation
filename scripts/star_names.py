"""Bayer star names with Greek letters, for display only: ``greek("kappa Vel")`` is "κ Vel".

Slugs and file names stay ASCII (kap_vel, omi_leo); the page generators apply this to the text they write.
A letter is replaced only when a space and a constellation abbreviation (case as the IAU writes it) follow it, so
ordinary words and slugs are left alone.
"""

import re

LETTERS = {
    "alpha": "α", "alf": "α", "beta": "β", "bet": "β", "gamma": "γ", "gam": "γ", "delta": "δ", "del": "δ",
    "epsilon": "ε", "eps": "ε", "zeta": "ζ", "zet": "ζ", "eta": "η", "theta": "θ", "tet": "θ", "iota": "ι",
    "iot": "ι", "kappa": "κ", "kap": "κ", "lambda": "λ", "lam": "λ", "mu": "μ", "nu": "ν", "xi": "ξ", "ksi": "ξ",
    "omicron": "ο", "omi": "ο", "o": "ο", "pi": "π", "rho": "ρ", "sigma": "σ", "sig": "σ", "tau": "τ",
    "upsilon": "υ", "ups": "υ", "phi": "φ", "chi": "χ", "psi": "ψ", "omega": "ω", "ome": "ω",
}
CONSTELLATIONS = (
    "And Ant Aps Aql Aqr Ara Ari Aur Boo Cae Cam Cap Car Cas Cen Cep Cet Cha Cir CMa CMi Cnc Col Com CrA CrB Crt Cru "
    "Crv CVn Cyg Del Dor Dra Equ Eri For Gem Gru Her Hor Hya Hyi Ind Lac Leo Lep Lib LMi Lup Lyn Lyr Men Mic Mon Mus "
    "Nor Oct Oph Ori Pav Peg Per Phe Pic PsA Psc Pup Pyx Ret Scl Sco Sct Ser Sex Sge Sgr Tau Tel TrA Tri Tuc UMa UMi "
    "Vel Vir Vol Vul"
).split()
SUPERSCRIPT = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
_NAME = re.compile(r"\b((?i:" + "|".join(sorted(LETTERS, key=len, reverse=True)) + r"))(\d?) (" + "|".join(CONSTELLATIONS) + r")\b")


def _sub(m):
    letter, digit, con = m.groups()
    return LETTERS[letter.lower()] + digit.translate(SUPERSCRIPT) + " " + con


def greek(text):
    """``text`` with every Bayer name (spelled-out or abbreviated letter, then a constellation) in Greek."""
    return _NAME.sub(_sub, text)
