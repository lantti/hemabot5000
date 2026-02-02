import numpy as np
import random
# Various Bolognese fencing terms mostly following Giovanni dall'Agocchie
guards = {"ga":"Guardia Alta", "gac":"Guardia d'Alicorno", "ge":"Guardia d'Entrare", "gf":"Guardia di Faccia", "gt":"Guardia di Testa",
          "gcls":"Coda Lunga Stretta", "gcla":"Coda Lunga Alta", "gcll":"Coda Lunga Larga", "gcld":"Coda Lunga Distesa",
          "gpfs":"Porta di Ferro Stretta", "gcpf":"Chingiara Porta di Ferro", "gpfa":"Porta di Ferro Alta", "gpfl":"Porta di Ferro Larga",
          "gspr":"Guardia Sopra il Braccio", "gsot":"Guardia Sotto il Braccio"}
          
cuts = {"mf":"Mandritto Fendente", "msq":"Mandritto Sgualimbro", "mt":"Mandritto Tondo", "mri":"Mandritto Ridoppio", "mtra":"Mandritto Tramazzone",
        "fd":"Falso Dritto", "mm":"Mezzo Mandritto", "rf":"Riverso Fendente", "rsq":"Riverso Sgualimbro", "rt":"Riverso Tondo",
        "rri":"Riverso Ridoppio", "rtra":"Riverso Tramazzone", "fm":"Falso Manco"}
        
thrusts = {"im":"Imbroccata", "st":"Stoccata", "pr":"Punta Riversa"}

# The practice sequence commonly called "Passeggiare nelle Guardie" from dall'Agocchie (1572)
passeggiare = ["rt", "rsq", "gcls", "fd", "msq", "gcpf", "mtra", "gpfs", "fm", "rsq", "gcla", "rri", "gac", "im", "gpfs", "fm", "rsq", "gcla",
               "mtra", "gpfs", "mtra", "gcpf", "fm", "rsq", "gcls"]

terms = guards | cuts | thrusts

# Numerical order for the term keys
termList = list(terms.keys())
termIndex = {termList[i]:i for i in range(len(termList))}
indexTerm = {i:t for t, i in termIndex.items()}

def softmax(a):
	exps = np.exp(a)
	tot = np.sum(exps)
	return exps / tot

def fit(seq):
	cmat = np.zeros((len(terms),len(terms)))
	pmat = np.zeros(cmat.shape)
	for i in range(len(seq)-1):
		cmat[termIndex[seq[i]]][termIndex[seq[i+1]]] += 1
	for i,row in enumerate(cmat):
		pmat[i] = softmax(row)
	return (cmat, pmat)

def step(term, pmat):
	return random.choices(termList, weights=pmat[termIndex[term]], k=1)[0]

def sequence(n, term, pmat):
	seq = []
	nextTerm = term
	for _ in range(n):
		nextTerm = step(nextTerm, pmat)
		seq.append(terms[nextTerm])
	return seq
