import numpy as np
# Modellmodellierungsfaktoren
anzahl = 1000
faktor=0.8
trainingsanteil=0.8

datensatz = {}
key = np.random.rand(anzahl)
i = 0
farbe=""
while(i < anzahl):
    if np.random.rand() < faktor:
        farbe="Blau"
    else:
        farbe="Rot"
    datensatz[i]=[key[i],farbe]
    i += 1
print(datensatz)
print("*" * 30)

trainingsdaten = {}
testdaten = {}
i = 0
while(i < anzahl*trainingsanteil):
    trainingsdaten[i]=datensatz[i]
    i += 1
print(trainingsdaten)
print("*" * 30)
while(i < anzahl):
    testdaten[i]=datensatz[i]
    i += 1
print(testdaten)
print("*" * 30)    
def beurteileWerte():
    fehler=0
    rotAnzahl=0
    rotWert=0
    blauAnzahl=0
    blauWert=0
    for a in trainingsdaten:
        #print(trainingsdaten[a][0], trainingsdaten[a][1])
        #print("*" * 30) 
        if trainingsdaten[a][1]=="Rot":
            rotAnzahl+=1
            rotWert+=trainingsdaten[a][0]
        else:
            blauAnzahl+=1
            blauWert+=trainingsdaten[a][0]
    rotIndikator=rotWert/anzahl
    blauIndikator=blauWert/anzahl
 
    print("Rot", rotAnzahl, rotWert, rotIndikator)
    print("Blau", blauAnzahl, blauWert, blauIndikator)
    for a in testdaten:
        if np.absolute(testdaten[a][0]-rotIndikator) < np.absolute(testdaten[a][0]- blauIndikator):
            vermutung="Rot"
        else:
            vermutung="Blau"
        #print("Vermutete Farbe:", vermutung, "Wirkliche Farbe:", testdaten[a][1], "Vermutung stimmt:", vermutung==testdaten[a][1])
        if vermutung!=testdaten[a][1]:
            fehler+=1
    print("Anzahl fehlerhafter Vermutungen",fehler, "Fehlerquote:", fehler/(anzahl*(1-trainingsanteil)))

beurteileWerte()
