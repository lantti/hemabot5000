import sklearn.neural_network as sklnn
import sklearn.linear_model as skll
import random
import math

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
termKeyOrder = list(terms.keys())

# One Hot encoding of the terms
oneHot = {k: [0]*n + [1] + [0]*(len(terms)-1-n) for n, k in enumerate(termKeyOrder)}


train_X = [oneHot[k] for k in passeggiare] 
train_y = train_X[1:] + [train_X[-1]]

# Simple NN
net = sklnn.MLPRegressor(hidden_layer_sizes=[25], solver='lbfgs', max_iter=2000)
net.fit(train_X, train_y)

# Linear regression for comparison
linreg = skll.LinearRegression()
linreg.fit(train_X, train_y)

regressor = net

# I couldn't figure out how set output activation to softmax on MLPClassifier,
# so I roll my own and use  MLPRegressor instead
def softmax(lst):
	exps = [math.exp(z) for z in lst]
	tot = sum(exps)
	return [e / tot for e in exps]


# Run the regressor prediction one or more times to get a sequence
# of terms. The prediction output is tested against random numbers
# to get a random term roughly corresponding to the output distribution
def step(key, steps=1):
	nextKey = key
	seq = []
	for _ in range(steps):
		pr = softmax(regressor.predict([oneHot[nextKey]])[0])
		nextKey = random.choices(termKeyOrder, weights=pr, k=1)[0]
		seq.append(nextKey)
	return seq
	
# Get a sequence in human readable format using the function "step"
def sequence(n, key=""):
	if key == "":
		key = random.choice(termKeyOrder)
	seq = step(key, n)
	return [terms[k] for k in seq] 
