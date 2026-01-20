import numpy as np
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class AminoAcid:
    """Représente un acide aminé"""
    name: str
    code_3: str
    code_1: str


# Liste des acides aminés communs
AMINO_ACIDS = [
    AminoAcid("Alanine", "Ala", "A" ),
    AminoAcid("Arginine", "Arg", "R" ),
    AminoAcid("Asparagine", "Asn", "N" ),
    AminoAcid("Acide aspartique", "Asp", "D" ),
    AminoAcid("Cystéine", "Cys", "C"),
    AminoAcid("Acide glutamique", "Glu", "E" ),
    AminoAcid("Glutamine", "Gln", "Q"),
    AminoAcid("Glycine", "Gly", "G" ),
    AminoAcid("Histidine", "His", "H"),
    AminoAcid("Isoleucine", "Ile", "I" ),
    AminoAcid("Leucine", "Leu", "L" ),
    AminoAcid("Lysine", "Lys", "K" ),
    AminoAcid("Méthionine", "Met", "M"),
    AminoAcid("Phénylalanine", "Phe", "F"),
    AminoAcid("Proline", "Pro", "P"),
    AminoAcid("Sérine", "Ser", "S"),
    AminoAcid("Thréonine", "Thr", "T"),
    AminoAcid("Tryptophane", "Trp", "W"),
    AminoAcid("Tyrosine", "Tyr", "Y"),
    AminoAcid("Valine", "Val", "V")
]

#Liste pour déterminer si un AA est deutérable (True) ou non (False)
restrictions = [
        True,   # Ala
        False,  # Arg
        True,   # Asn
        True,   # Asp
        False,  # Cys
        False,  # Glu
        True,   # Gln
        True,   # Gly
        False,  # His
        True,   # Ile
        True,   # Leu
        True,   # Lys
        True,   # Met
        True,   # Phe
        True,   # Pro
        True,   # Ser
        True,   # Thr
        True,   # Trp
        True,   # Typ
        True,   # Val
    ]