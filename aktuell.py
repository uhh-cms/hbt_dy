
from dataclasses import dataclass, field

@dataclass
class Smartphone:
    # Pflichtfelder (müssen beim Erstellen angegeben werden)
    modell: str
    marke: str
    
    # Felder mit Standardwerten (optional beim Erstellen)
    speicher_gb: int = 128
    ist_eingeschaltet: bool = False
    
    # Ein Feld, das eine Liste ist (erfordert default_factory)
    installierte_apps: list = field(default_factory=lambda: ["Kamera", "Einstellungen"])

    def einschalten(self):
        self.ist_eingeschaltet = True
        print(f"Das {self.modell} wurde eingeschaltet.")
    def ausschalten(self):
        self.ist_eingeschaltet = False
        print(f"Das {self.modell} wurde ausgeschaltet.")


from IPython import embed; embed(header="MESSAGE Line 22 | File: aktuell.py")

# --- Nutzung ---

mein_handy = Smartphone("Pixel 10","Google",speicher_gb=256,ist_eingeschaltet=True)
# 1. Eine Instanz erstellen (Objekt)
dein_handy = Smartphone(modell="iPhone 15", marke="Apple", speicher_gb=256)

# 2. Eine zweite Instanz erstellen (nutzt Standardwerte)
sein_handy = Smartphone(modell="Pixel 8", marke="Google")

from IPython import embed; embed(header="MESSAGE Line 36 | File: aktuell.py")

# 3. Zugriff auf die Daten
print(f"Mein Handy hat {mein_handy.speicher_gb} GB.")
print(f"Deine Apps: {dein_handy.installierte_apps}")

# 4. Eine Methode aufrufen
mein_handy.einschalten()
