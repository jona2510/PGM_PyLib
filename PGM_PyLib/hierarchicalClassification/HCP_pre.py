"""
This code belongs to the Probabilistic Graphical Models Python Library (PGM_PyLib)
	PGM_PyLib: https://github.com/jona2510/PGM_PyLib

Check the "PGM_PyLib Manual vX.X.pdf" to see how the code works.

The PGM_PyLib is distributed under the GNU public license v3.0.

Code author: Jonathan Serrano-Pérez
"""

#funciona con los *data_nm.arff y *header_nm.arff 
#creados previamente por simple_pruning.py
import numpy as np
import random
import os
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix as cm
from sklearn.naive_bayes import GaussianNB as gnb
from sklearn.ensemble import RandomForestClassifier as rfc
import junctiontree.beliefpropagation as bp
import junctiontree.junctiontree as jt
import math
from time import time
import sys

import PGM_PyLib.naiveBayes as nb
import copy



#verifica si un nodo es raiz
#recibe root, y la matriz estructura 
def isroot(root,mest,dic):
	if(sum(mest[:,dic[root]])>0):
		return False
	else:
		return True
	#x=0


#recibe root, y la matriz estructura 
def isleaf(root,mest,dic):
	if(sum(mest[dic[root],:])>0):
		return False
	else:
		return True
	#x=0

#funcion de ancestros recurisiva 
# NO modificar para dag, probablemente minimo
# root - nodo al que se le sacaran sus padre, es indice
# anc  lista de ancestros
# mest grafo 
# dic
def ancestor_rec(root,mest,anc):
	x=2
	l=[]	#lista de ancestros
	for i in range(len(mest)):
		if(mest[i,root]==1):
			#aux=[x[0] for x in dic.items() if x[1]==i]
			if(not i in anc):
				l.append(i)
				anc.append(i)
				

	for x in l:	#para cada padre "diferente" (que no este previamente en anc) obtener sus ancestros
		ancestor_rec(x,mest,anc)

# root es indice
def ancestors(root,mest):
	x=1
	l=[]	#lista de ancestros
	for i in range(len(mest)):
		if(mest[i,root]==1):
			#aux=[x[0] for x in dic.items() if x[1]==i]
			#l.append(aux[0])
			l.append(i)

	anc=[x for x in l]
	for x in l:	#para cada padre obtener sus ancestros
		ancestor_rec(x,mest,anc)
	

	return anc


#root nodo al que se le construira el clasificador
# train, test datos/atributos
# cltr, clte clases de train y test respectivamente
# dic diccionario
# mest matriz de estructura
# dclas diccionario que contiene las salidas de los clasificadores
# dclas_prob diccionario que contiene las probabilidades salidas de los clasificadores
# classes_order las cases en orden (sirve para dclas_prob)
# smoothing suavizado (True, False)
def build_cpt_obs(root,train,test,cltr,clte,dic,dic_inv,mest,dclas,dclas_prob,classes_order,smoothing, baseClassifier=rfc()):
	#necesito los valores reales		de test
	#necesito los valores predecidos	de test
	#necesito el modelo del clasificador
	h=0

	#******************** TRAIN *********************************
	#datos y clases de train 
	att_aux=[]
	cl_aux=[]

	#dtr=[]
	#dcl=[]
	#ejemplos positivos
	band=isroot(root,mest,dic)
	f_pos=[1 for i in range(0,len(mest)) if(mest[i,dic[root]]==1) ]	#creo un vector que incluye la cantidad de padres, que son positivos

	for i in range(0,len(cltr)):
		#spl=cltr[i].split(root)
		#if(spl[0]==""):
		if(cltr[i,dic[root]]==1):
			if(band):	#si es raiz, simplemente se agregarn los atributos
				#dtr.append(att[i]+",1")
				att_aux.append(train[i])				
			else:
				#si no es raiz, se agrega la salida de sus padres, en este caso siempre es uno
				#****reanalizar dado que todas las 
				att_aux.append(np.concatenate([train[i], f_pos ]) )
				
			#la lcase positiva se concatena normalmente
			cl_aux.append(1)
				
	#nodo padre, en dag puede haber varios padres
	#spl2=root.split("/")
	#spl2.pop() 			#elimino el ultimo "nodo", para quedarme con el padre
	#p="/".join(spl2)	#creo el nodo padre
	p=[i for i in range(0,len(mest)) if(mest[i,dic[root]]==1)]	# saco los padres de root

	ndtr=len(att_aux)		#len(dtr)
	max_nb=round(ndtr*1.0)		#"porcentaje" del numero de hermanos
	c=0
	#ejemplos negativos
	#cambiar por algo mas sofisticado
	for i in range(0, len(cltr)):
		#spl=cltr[i].split(root)
		#if(spl[0]!=""):
		if(cltr[i,dic[root]]==0):	#si no esta asociado a root
			if(band):		#si es raiz, se agregan los ejemplos negativos (que correponden directamente a los de sus hermanos)
				#dtr.append(att[i]+",0")
				att_aux.append(train[i])
				cl_aux.append(0)
				c=c+1
				if(c>=ndtr):
					break
			else:
				#reviso si tienen el mismo padre;
				#spl2=cltr[i].split(p)
				#if(spl2[0]==""):	#solo si tienen el mismo padre, lo agrego
				if(len([j for j in p if(cltr[i,j]==1) ])>0):	#si comparten almenos un padre, lo agrego	- costoso
					#dtr.append(att[i]+",0")
					real_ext=[cltr[i,x] for x in p]
					att_aux.append(np.concatenate([train[i],real_ext] ))					

					cl_aux.append(0)
					c=c+1
					if(c>=max_nb):	#era ndtr, se cambia por el porcentaje
						break

	if(c<ndtr and not band):		#si no hubo suficientes ejemplos negativos de sus hermanos
		for i in range(0, len(cltr)):
			#spl=cltr[i].split(root)
			#if(spl[0]!=""):
			if(cltr[i,dic[root]]==0):	#si no esta asociado a root
				# lo del if ya no va, porque en caso de agregarlo, se duplicarian ejemplo de arriba
				#if(isroot(root,mest,dic)):
				#	#dtr.append(att[i]+",0")
				#	att_aux.append(train[i])
				#	cl_aux.append(0)
				#	c=c+1
				#	if(c>=ndtr):
				#		break
				#else:

				#reviso que el padre sea diferente
				#spl2=cltr[i].split(p)
				#if(spl2[0]!=""):	#solo si el padre es diferente se agrega
				if(len([j for j in p if(cltr[i,j]==1) ])==0):	#si no comparten padres, lo agrego	- costoso
					#att_aux.append(np.concatenate([train[i],[1]] ))
					#else:
					real_ext=[cltr[i,x] for x in p]
					att_aux.append(np.concatenate([train[i],real_ext] ))

					cl_aux.append(0)
					c=c+1
					if(c>=ndtr):
						break
				
				
								
	#print("hay "+str(c)+" de clase negativa")
	#print("hay "+str(ndtr)+" de clase positiva")

	if(c<1 or ndtr<1):
		print("ERROR: en calculo de cpt_obs")
		print("No ha ejemplos suficientes")
		print("hay "+str(c)+" de clase negativa")
		print("hay "+str(ndtr)+" de clase positiva")
		#exit()

	# se convierten a array nnumpy para los clasificadores de sklearn
	att_aux=np.array(att_aux)
	cl_aux=np.array(cl_aux)	

	#******************** TEST ***********************************

	#aqui pregunto si es root, si es root se deja normal, si no, se agrega como atributo la salida de su padre, en dclas
	dtr_t=[]
	dtr_c=clte[:,dic[root]]	#[]
	cpos=0
	cneg=0

	if(isroot(root,mest,dic)):
		dtr_t=test
		##ejemplos positivos y negativos 
		#for i in range(0,len(clte)):
		#	spl=clte[i].split(root)
		#	if(spl[0]==""):
		#		#dtr_t.append(att_t[i]+",1")
		#		dtr_t.append(test[i])
		#		dtr_c.append(1)
		#		cpos=cpos+1
		#	else:
		#		#dtr_t.append(att_t[i]+",0")
		#		dtr_t.append(test[i])
		#		dtr_c.append(0)
		#		cneg=cneg+1
	else:	#si no es nodo raiz, estoy suponiendo un solo padre
		#ejemplos positivos y negativos 
		for i in range(0,len(clte)):
			ex_real=[]
			for x in p:
				#if(not dic_inv[x] in dclas.keys()):
				#	print("no esta en dclas: "+dic_inv[x])
				#	exit()
				ex_real.append(dclas[dic_inv[x]][i])		#	ESTA MAL *************************+ corregido
			dtr_t.append(np.concatenate([test[i],ex_real]) )

			#spl=clte[i].split(root)
			#if(spl[0]==""):
			#	#dtr_t.append(att_t[i]+",1")
			#	
			#	dtr_t.append(np.concatenate([test[i],[dclas[p][i] ]]) )		#concateno la salida del padre para el i-esimo elemento
			#	dtr_c.append(1)
			#	cpos=cpos+1
			#else:
			#	#dtr_t.append(att_t[i]+",0")
			#	dtr_t.append(np.concatenate([test[i],[dclas[p][i] ]]) )
			#	dtr_c.append(0)
			#	cneg=cneg+1
		dtr_t=np.array(dtr_t)	# se convierten a array nnumpy para los clasificadores de sklearn
	#print("test::  pos: "+str(cpos)+", neg:"+str(cneg))

	# se convierten a array nnumpy para los clasificadores de sklearn
	#dtr_t=np.array(dtr_t)
	#dtr_c=np.array(dtr_c)	


	#********************* Clasificacion ******************************
	clf=copy.deepcopy(baseClassifier)		#gnb() , SVC() , rfc()
	clf.fit(att_aux, cl_aux)

	predictions=clf.predict(dtr_t)

	if(len(clf.classes_)>1):
		if(classes_order[0]==0 and classes_order[1]==0):	#si no se han obtenido las clases [0,1]
			classes_order[0]=clf.classes_[0]#.copy()
			classes_order[1]=clf.classes_[1]
			#print("classes_order")
			#print(classes_order)
			#input("pausa +++")
		else:	#si ya estan configuradas
			#verificar que tienen el mismo orden:
			if(classes_order[0]!=clf.classes_[0] or classes_order[1]!=clf.classes_[1]):
				print("ERROR: el orden de las clases es diferente")
				exit()
		dclas_prob[root]=clf.predict_proba(dtr_t)
	else:
		if(classes_order[0]==0 and classes_order[1]==0):	#si no se han obtenido las clases [0,1]
			classes_order[0]=0#clf.classes_[0]#.copy()
			classes_order[1]=1#clf.classes_[1]
		#else:	#si ya estan configuradas
		#	#verificar que tienen el mismo orden:
		#	#if(classes_order[0]!=clf.classes_[0] or classes_order[1]!=clf.classes_[1]):
		#	#	print("ERROR: el orden de las clases es diferente")
		#	#	exit()
		dclas_prob[root]=np.zeros((len(test),2))#clf.predict_proba(dtr_t)
		for i in range(len(test)):
			dclas_prob[root][i][classes_order.index(clf.classes_[0])]=1.0	#unica clase		

	
	#se agregan las predicciones al diccionario de clasificacion
	dclas[root]=predictions

	#se agregan las probabilidades al diccionario de probabilidades
	#dclas_prob[root]=clf.predict_proba(dtr_t)

	#for i in range(5):
	#	print("prediction: ",predictions[-(i+1)])

	#print("Accuracy "+root+":"+str(clf.score(dtr_t,dtr_c)))
	cmat=cm(dtr_c,predictions)
	if(sum(dtr_c)==len(dtr_c) and sum(predictions)==len(predictions)):	#todo es de clase positiva
		tn=0
		fp=0
		fn=0
		tp=len(predictions)
	elif(sum(dtr_c)==0 and sum(predictions)==0):	#todo es de clase negativa
		tn=len(predictions)
		fp=0
		fn=0
		tp=0		
	else:
		tn, fp, fn, tp = cm(dtr_c,predictions).ravel()
	#print(cmat)
	#print("tn, fp, fn, tp:")
	#print(tn, fp, fn, tp)
	#print("")
	#print("cpt observaciones "+root+": ")

	#sin suavizado laplaciano	no suma en f,f
	if(smoothing):	# si suavizado
		fp=fp+1
		tn=tn+1
		fn=fn+1
		tp=tp+1

	#tabla de 4 valores
	#		hijo:	false	true
	#padre: false
	#padre: true 
	table=np.zeros((2,2))	
	table[0,0]=	float(tn) #/(float(tn)+float(fp))
	table[1,1]=	float(tp) #/(float(fn)+float(tp))
	table[1,0]=	float(fn) #/(float(fn)+float(tp))
	table[0,1]=	float(fp) #/(float(tn)+float(fp))
	#print(table)

	return table,predictions,dtr_c




def add_cpt_rec(index,indices,table):

	if(index==(len(indices)-1) ):
		table[indices[index]]=table[indices[index]]+1
	else:
		add_cpt_rec(index+1,indices,table[indices[index]])
	

# indices arreglo de indices de la posicion a la que se sumara uno
# table la tabla a la que se le sumara
def add_cpt(indices,table):
	x=0

	if(len(indices)<2):
		print("Error en el indice")
		exit()
	add_cpt_rec(x+1,indices,table[indices[x]])


def gen_comb_sm_rec(index,indices,table):	
	
	if(index==(len(indices)-1)):
		indices[index]=0	#lo coloco a cero para hacer la suma correcta
		if(sum(indices)==(len(indices)-1)):	#si todos los padres son uno
			#suaviza y normalizo 
			table[0]=table[0]+1
			table[1]=table[1]+1
			a0=table[0]/sum(table)
			a1=table[1]/sum(table)
			table[0]=a0		
			table[1]=a1
		elif(sum(indices)==0):
			# forzo las probabilidades 
			table[0]=0.99
			table[1]=0.01
		else:
			# forzo las probabilidades 
			table[0]=0.5
			table[1]=0.5
			
	else:		
		for i in range(2):	# dos valores que puede tomar
			indices[index]=i
			gen_comb_sm_rec(index+1,indices,table[i])		


# nele es el numero de padres
def gen_comb_sm(nele,table):
	x=0
	
	arr_index=np.zeros(nele+1).astype(int)

	if(len(arr_index)<2):
		print("Error en el indice")
		exit()

	for i in range(2):	# dos valores que puede tomar
		arr_index[x]=i
		gen_comb_sm_rec(x+1,arr_index,table[i])		


# root nodo al que se le calculara su cpt
# mest la jerarquica
# dic, diccionario con los nombre de los nodos y su indice correspondiente
# cltr, clases del train
def cpt(root,mest,dic,cltr):

	fathers=[]		

	#de momento se supone que cada nodo tiene a lo mas un padre
	if(sum(mest[:,dic[root]])<1):
		#no tiene padre, calcular las marginales
		table=np.zeros((2,2))	#posicion 0: false, 1: true	
		for i in range(0,len(cltr)):
			#spl=cltr[i].split(root)
			#if(spl[0]==""):		#si es
			if(cltr[i,dic[root]]==1):	# si i esta asociado a root
				table[1,1]=table[1,1]+1.0
			else:				#no es
				table[1,0]=table[1,0]+1.0
		#suavizado laplaciano
		table[1,0]=table[1,0]+1.0
		table[1,1]=table[1,1]+1.0
		table[0,0]=1
		table[0,1]=0

		table[1]=table[1]/sum(table[1])
	else:	# tiene VARIOS PADRES

		#tabla de 4 valores
		#		hijo:	false	true
		#padre: false
		#padre: true 
		#table=np.zeros((2,2))	


		fathers=[x for x in range(len(mest)) if(mest[x,dic[root]]==1)]
		table=np.zeros([2 for x in range(len(fathers)+1)])	#crea la matriz para la Tabla de Probabilidad Condicional

		#f=[x[0] for x in dic.items() if x[1]==fathers[0]]
		arr_index=np.zeros(len(fathers)+1).astype(int)	#lleva la posicion de la matriz a modificar
		
		for i in range(len(cltr)):
			#spl=cltr[i].split(f[0])
			#if(spl[0]==""):		#si es padre
			#	#table[1]=table[1]+1.0
			#	spl2=cltr[i].split(root)
			#	if(spl2[0]==""):	#si es hijo
			#		table[1,1]=table[1,1]+1
			#	else:
			#		table[1,0]=table[1,0]+1
			for j in range(len(fathers)):
				arr_index[j]=cltr[i,fathers[j]]
			arr_index[-1]=cltr[i,dic[root]]		#si las clases (cltr) estan bien, solo cuando todos los padres son 1, root tambien lo PUEDE ser

			# llamar a metodo recursivo para modificar la tpc
			if(sum(arr_index)>=len(fathers)):	#ya que en otro caso no es necesario
				add_cpt(arr_index,table)					#costoso y lento

		#forzar todas las probabilidades a 0-1, P(y=1|pa(y)!=1), todas las combinaciones exepto cuando pa(y)=1
		#necesito generar todas las combinaciones
		gen_comb_sm(len(fathers),table)	#combinacion y suavizado		

		#P(hijo=falso|padre=falso)=1
		#suavizado laplazaciano
		#table[0,0]=1
		#table[1,0]=table[1,0]+1
		#table[1,1]=table[1,1]+1

		#table=table/sum(sum(table))
		#table[1]=table[1]/sum(table[1])
				
			#else:				#no es
			#	table[0]=table[0]+1.0
			
			
	#lib if(len(fathers)>1):
	#lib 	print("cpt "+root+" (tam= "+str(len(fathers))+"): ")
	#lib 	print(table)
	#lib 	print("")

	return table




def add_factor_values3(x,factors,values,cptn,mest,dic,dic_inv,dicpos):
	aux=[]
	dicpos[x]=len(dicpos)
	if(sum(mest[:,dic[x]])>0):	#Tiene padres
		for i in range(len(mest)):
			if(mest[i,dic[x]]==1):
				#f=[x[0] for x in dic.items() if x[1]==i]
				#aux.append(f[0])
				aux.append(dic_inv[i])
		aux.append(x)
		factors.append(aux)
		values.append(cptn[dic[x]])
		#		break
	else:						#no tiene padre
		#los nodos raices estan ligados con main_root
		aux.append("main_root")
		aux.append(x)
		factors.append(aux)
		values.append(cptn[dic[x]])

	#esto es (bpp)rofundidad, quiza para dag convenga a lo ancho
	#for i in range(len(mest)):
	#	if(mest[dic[x],i]==1):
	#		f=[x[0] for x in dic.items() if x[1]==i]
	#		add_factor_values3(f[0],factors,values,cptn,mest,dic,dicpos)


# sirve para dag, bueno, 
# anc_leaves diccionario con los ancestros de cada nodo hoja
# out_leaves los vectores de clase asociados a los ancestros de cada nodo hoja
# probm las probabilidades de que una instancia este asociada a cada nodo
def score_sp(anc_leaves,out_leaves,probm,dic):
	x=0	#dic_anc_leaves, dic_out_leaves	
	prom_max=-np.inf
	out_leaf=np.nan	#no hay nada

	for x in anc_leaves.keys():		#para cada nodo hoja
		prom_l=probm[dic[x]]		# el nodo hoja "x" no esta en los ancestros de x, por eso es el primer valor
		for z in anc_leaves[x]:		# para cada ancestros
			prom_l=prom_l+probm[z]	
		prom_l=prom_l/(float(len(anc_leaves[x])) +1.0)

		if(prom_l>prom_max):
			prom_max=prom_l
			out_leaf=out_leaves[x]

	return out_leaf


def score_glb_rec(ind,mest,probm,weights,path):
	sclocal=weights[ind]*np.log10(probm[ind])
	path[ind]=1

	s_p=0.0		#puntaje de los padres
	for i in range(len(mest)):
		if(mest[i,ind]==1):	#si tiene padre
			s_p=s_p+score_glb_rec(i,mest,probm,weights,path)

	return (sclocal+s_p)


#calcula la puntuación de cada trayectoria, segun tesis mellinali	(deberia funcionar en DAGS)
# leafs conjunto de nodos hoja
# mest estructura del arbol/dag
# probm probabilidades "positivas" de cada nodo
# n_f numero de padres de cada nodo
# weights pesos de cada nodo
def score_glb(leafs,mest,probm,dic,n_f,weights):
	scmax=-math.inf		#puntuación maxima
	path=np.zeros(len(mest))

	
	prm=probm/n_f	#divide la probabilidad entre el numero de padres (esto para DAGS, en arboles no afecta)

	for x in leafs:	
		# considero trayectorias, aquellas que llegan a nodo hoja	
		# analizar si una trayectoria a nodo hoja puede ser una de dag
		# es decir, una trayectoria puede tener dos+ caminos que lleguen al mismo lugar
		p_aux=np.zeros(len(mest))
		sc=score_glb_rec(dic[x],mest,prm,weights,p_aux)
		if(sc>scmax):
			scmax=sc
			path=p_aux
	
	return path		#regresa el vector con los ceros y unos
	

# calcula el peso de cada nodo (se usa en GLB)
def nodes_weight(mest):
	w=np.zeros(len(mest))
	visited=[]

	cola=[]
	
	#la busqueda debe hacerce tipo bpa, asegurando que los padres de cada nodo han sido visitado primero (muy importante en DAG)

	#inserto "raices" (niveles 1), la main_root sera la de nivel 0
	for i in range(0,len(mest)):
		if(sum(mest[:,i])==0):
			cola.append(i)		#inserto el indice, no conozco el nombre del indice

	#nodes_weight_rec(w,visited,cola,mest)

	while(len(cola)>0):
		#print(cola)
		aux=cola.pop(0)	
		p=[]
		#verificar si es root
		if(sum(mest[:,aux])==0):
			w[aux]=1.0		#parece estar bien el 1	(1, segun tesis de mellani)
			visited.append(aux)
		else:	#si no es root, verificar que sus padre han sido visitados primero
			ban=True	#bandera de que sus padres han sido visitados
			for i in range(0,len(mest)):
				if(mest[i,aux]==1 ):
					if(i in visited):
						p.append(w[i])					
					else:
						ban=False
						break
			if(ban):	#si sus padres han sido visitados
				w[aux]=sum(p)/len(p)+1.0	#promedio de los niveles de los padres
				visited.append(aux)
			else:	#sus padres no han sido visitados
				#se regresa a la cola
				cola.append(aux)
		#agrego a la cola sus hijos
		for i in range(0,len(mest)):
			if(mest[aux,i]==1 and not i in cola and not i in visited):
				cola.append(i)

	#hasta este punto w contiene los niveles de los nodos
	#por lo que obteniendo el valor maximo de w, obtengo el valor del nivel maximo
	ml=max(w)				#max_level(mest)
	#print("ml: "+str(ml))
	for i in range(0,len(mest)):
		w[i]=1.0-w[i]/(ml+1)

	return w,ml

			
#devuelve el numero de hermanos de cada nodo mas uno
def n_siblings(mest):
	N=np.zeros(len(mest))	#numero de hermanos de cada nodo
	
	for i in range(0,len(mest)):
		c=0
		if(sum(mest[:,i])==0):	#si es nodo raiz
			#cuentan todos los nodos raiz

			for j in range(0,len(mest)):
				if(sum(mest[:,j])==0):
					c=c+1
			N[i]=c		#no se suma uno, porque ya se conto uno de mas (el mismo nodos), en el for de arriba
		else:
			#si no es nodo raiz
			#contar sus hermanos, de cada padre (DAG)
			for j in range(0,len(mest)):
				if(mest[j,i]==1):	#si j es padre, entonces contar los hijos de j excepto i
					for k in range(0,len(mest)):
						if(mest[j,k]==1 and k!=i):
							c=c+1

			N[i]=c+1	#hermanos mas uno
	return N

#devuelve el numero de padres de cada nodo
def n_fathers(mest):

	n_f=np.zeros(len(mest))
	
	for i in range(0,len(mest)):
		x=sum(mest[:,i])
		if(x==0):	#si es nodo raiz, suma uno (tiene a main_root como padre)
			n_f[i]=1.0
		else:
			n_f[i]=x

	return n_f


def HCP(header_in, train_in, test_in, tscore="SP", type_prop="all_probabilities", baseClassifier=rfc()):
	"""
	header_in= file which contains the header in arff-type format 
	train_in= file which contains data for trainig (which follow the indication of the header)
	test_in= file which contains data for testing
	name_out=sys.argv[4]

	tscore=sys.argv[5]	#"SP"	#tipo de puntuacion para las trayectorias:	GLB , SP
	type_prop=sys.argv[6]	#"all_probabilities"		#conocimiento para la propagacion (red bayesiana)  all_probabilities , just_positives
	"""

	# necesito los siguientes archivos: header, train_data, test_data 

	start_time=time()

	#path="/home/jonathan/inaoe/PGM/tesis/datasets/datasets_GO/"	#jonathan , o-2016-037-p
	#path="/home/o-2016-037-p/inaoe/PGM/tesis/datasets/datasets_GO_mallinali/NMLNP_SP/"	#jonathan , o-2016-037-p

	#lib if(len(sys.argv)!=7):
	#lib 	#print("ERROR: you must execute: CRL_v3.py name_dataset m|nm SP|GLB all_probabilities|just_positives")
	#lib 	print("ERROR: you must execute: dag2_CRL_v3.py header_dataset train_data test_data name_out SP|GLB all_probabilities|just_positives")
	#lib 	exit()

	#lib header_in=sys.argv[1]
	#lib train_in=sys.argv[2]
	#lib test_in=sys.argv[3]
	#lib name_out=sys.argv[4]

	#dataset=sys.argv[1]	#"eisen_FUN"
	#monm=sys.argv[2]	#"m"		#mandatory o non-mandatory:		m , nm
	#lib tscore=sys.argv[5]	#"SP"	#tipo de puntuacion para las trayectorias:	GLB , SP
	#lib type_prop=sys.argv[6]	#"all_probabilities"		#conocimiento para la propagacion (red bayesiana)  all_probabilities , just_positives


	#path="/home/jonathan/inaoe/PGM/tesis/datasets/artificial/"
	#dataset="artificial2"	#era 1
	#pathsc=path+dataset+"/"+dataset+".full_" #"valid.arff"

	dic={}
	dic_inv={}
	cl_wop=""
	#with open(pathsc+"header_"+monm+".arff","r") as f:		# era nm
	with open(header_in,"r") as f:		# era nm
		ban=True
		while(ban):
			line=f.readline()

			spl=line.split("class")
			if(len(spl)>1):
				cl_wop=spl[1].strip()		#obtiene las clases sin procesamiento, se usara para sacar la estructura del DAG

			spl=line.split("@ORDEN")
			if(len(spl)>1):				
				#print("encontre class")
				spl2=spl[1].strip().split(",")
				c=0
				for i in range(0,len(spl2)):
					dic[spl2[i]]=c
					dic_inv[c]=spl2[i]
					c=c+1
				ban=False
				#line=spl[0]+"class {"+spl2[1]+"}\n"		
	f.close()
	nn=len(dic)		#numeros de nodos
	print("numero de etiquetas(nodos): "+str(nn))

	nd=(np.zeros(nn)).astype(int)	#numero de instancias por nodo
	mest=(np.zeros((nn,nn))).astype(int)	#matriz que representa la estructura del arbol/poliarbol/grafo

	#lib print("len dic: "+str(len(dic)))
	#lib print("len dic_inv: "+str(len(dic_inv)))

	#obtiene la estructura del DAG
	spl=cl_wop.split(",")
	for i in range(0,len(spl)):
		spl2=spl[i].split("/")
		if(spl2[0]!="root"):
			mest[dic[spl2[0] ],dic[spl2[1] ]]=1
		else:
			print("no se agrego root,*")

	roots=set()
	hojas=[]
	for i in range(0,len(mest)):
		if(sum(mest[:,i])==0):
			roots.add(dic_inv[i])
		if(sum(mest[i,:])==0):
			hojas.append(dic_inv[i])

	print("raices "+str(len(roots))+":")
	print(sorted(roots))

	print("hojas "+str(len(hojas))+":")
	print(sorted(hojas))

	"""
	roots=set()
	for x in sorted(dic.keys()):
		spl=x.split("/")
		roots.add(spl[0])

	print("raices:")
	print("num nodos raices: "+str(len(roots)))
	print(sorted(roots))

	for x in sorted(dic.keys()):
		#cada x indica una trayectoria
		spl=x.split("/")
		pl=spl[0]
		for i in range(0,len(spl)-1):	#era desde uno, pero parece que estaba mal
			pla=pl+"/"+spl[i+1]			#por lo de arriba se agrega el +1
			mest[dic[pl],dic[pla]]=1
			pl=pla
			
	hojasv=[]
	for i in range(0,nn):
		if(sum(mest[i])==0):
			hojasv.append(i)

	hojas=[x[0] for x in dic.items() if x[1] in hojasv]
	print("hojas:")
	print("num nodos hojas: "+str(len(hojas)))
	print(hojas)
	"""	
	###  se cargan los datos y se dividen en atributos/data y su clase (que mas bien indica su profundidad)
	data=[]
	classes=[]
	#with open(pathsc+"data_"+monm+".arff","r") as f:		# era nm
	with open(train_in,"r") as f:		# era nm
		for line in f:
			spl=line.strip().split(";")		#en dag separa los tributos de las clases con un ;
			#cl=	spl.pop()			#guarda la clase
			
			#data.append(",".join(spl))
			data.append(spl[0])		#agrego directaamente los atributos
			classes.append(spl[1].split(","))	#agrego las clases como cadenas de ceros y unos
	f.close()

	#carga datos de testa_data
	test=[]
	cltest=[]
	#with open(pathsc+"data_"+monm+".arff","r") as f:		# era nm
	with open(test_in,"r") as f:		# era nm
		for line in f:
			spl=line.strip().split(";")		#en dag separa los tributos de las clases con un ;
			#cl=	spl.pop()			#guarda la clase
			
			#data.append(",".join(spl))
			test.append(spl[0])		#agrego directaamente los atributos
			cltest.append(spl[1].split(","))	#agrego las clases como cadenas de ceros y unos
	f.close()

	print("Estrucutra de la jerarquia")
	print(mest)

	we,ml=nodes_weight(mest)	#recibo los pesos de los nodos en we, y el nivel maximo en ml
	print("pesos de los nodos")
	print(we)
	print("ml: "+str(ml))

	N=n_siblings(mest)
	print("Numero de hermanos mas uno, por nodo")
	print(N)

	n_f=n_fathers(mest)		#numero de padres, *los nodos raiz tienen a main_root como padre
	print("Numero de padres por nodo:")		#hecho para DAGS pero verificar
	print(n_f)

	dic_anc_leaves={}		#dicionario con los ancestros de leaves
	dic_out_leaves={}		#diccionario con los vectores de salida de cada nodo hoja

	#lib print("creando ancestros de hojas y sus vectores de clases")
	for x in hojas:
		dic_anc_leaves[x]=ancestors(dic[x],mest)
		dic_out_leaves[x]=np.zeros(len(mest)).astype(int)
		for z in dic_anc_leaves[x]:
			dic_out_leaves[x][z]=1
		dic_out_leaves[x][dic[x]]=1		# tambien esta asociado al nodo hoja
	
	#input("escribe algo y da enter '->")

	#*************+ intrudciendo sklearn para clasificar ************************************************
	
	#supongo que todos los atributos son numericos
	#lo cual es falso, analizar como convertir categoricos a numericos

	ndata=len(data)			#numero de datos
	if(ndata<5):
		print("Error!!!: few data")
		exit()

	natt=len(data[0].split(","))	#numero de atributos	

	#analizar datos faltantes, influye si es numerico o categorico
	#de momento se considera que se tienen todos los datos
	datafc=np.zeros((ndata,natt))

	for i in range(0,ndata):
		daux=data[i].split(",")
		for j in range(0,natt):
			#comprueba que no sea dato faltante
			if(daux[j]=="?" or daux[j]==""):
				datafc[i,j]==np.nan
			else:
				datafc[i,j]=float(daux[j])
	
	#analisis de datos daltantes para test
	testfc=np.zeros((len(test),natt))

	for i in range(0,len(test)):
		daux=test[i].split(",")
		for j in range(0,natt):
			#comprueba que no sea dato faltante
			if(daux[j]=="?" or daux[j]==""):
				testfc[i,j]==np.nan
			else:
				testfc[i,j]=float(daux[j])
	
	fold=0
	nfolds=1	#era 5 
	rang=float(ndata)/float(nfolds)		

	train=datafc		#creo un apuntador
	classtr=np.array(classes).astype(int)	#creo un apuntador

	test=testfc			#creo apuntador
	cltest=np.array(cltest).astype(int)		#creo otro apuntador

	nmeasures=7		#numero de medias de validacion programadas 
	res_dm=np.zeros((nmeasures,nfolds))		#matriz donde se guardaran los resultados de las diferentes medidas de validacion de los folds

	#print("train************************")
	#print(train)

	#print("test************************")
	#print(test)

	#por aqui iria un for para los n-folds
	for fold in range(0,nfolds): 		#*********************************
		tini=int(round(rang*fold))#instalar paquete en python
		tfin=int(round(rang*(fold+1)))

		#test=np.zeros((tfin-tini,natt))
		#cltest=np.zeros((tfin-tini,len(dic)))	#.astype(str)

		#copia de train a test
		#test[:,:]=train[:(tfin-tini),:]
		#cltest[:]=classtr[:(tfin-tini),:]
		#remover elementos de train y classtr
		#train=np.delete(train,np.s_[:(tfin-tini)],0)
		#classtr=np.delete(classtr,np.s_[:(tfin-tini)],0)

		#train=np.zeros((ndata-(tfin-tini),natt))

		#variable que guarda las salidas de los clasificadores
		dclas={}
		#variable que guarda las probabilidades salidas de los clasificadores
		dclas_prob={}
		# guarda el orden de las clases en dclas_prob
		c_order=[0,0]

		
		#variables para recuperar tablas condicionales (cpt)
		#observaciones y datos reales por nodo
		cptnodes=[]
		cptobs=[]
		obs=[]
		vreal=[]
		cptobs_aux=[]
		for i in range(len(mest)):
			cptnodes.append(0)
			cptobs_aux.append(0)
			cptobs.append(np.zeros((2,2)))
			obs.append(0)
			vreal.append(0)

		"""
		#en lo cola se lleva el orden en como se construiran los clasificadores (tipo bpa)
		cola=hojas.copy()
		while(len(cola)>0):
			x=cola.pop(0)		#extraigo el primer elemento de la cola

			#agrego los hijos de "x" a la cola
			#	comprobando que no esten en la cola
			for i in range(0,len(mest)):
				if(mest[dic[x],i]==1):
					#obtengo el nombre del nodo
					aux=[r[0] for r in dic.items() if(r[1]==i)]
					#agrego
					if(len(aux)>0):
						if(not aux[0] in cola):
							cola.append(aux[0])
					
			print("-------------------------------------------")
			print("llego: "+x)
			#rec2(x,data,classes,mest,pathsc+"header_nm.arff",ftest,ctest,mest,dic)
			#build_classifiers(x,data,classes,mest,pathsc+"header_nm.arff",ftest,ctest,mest,dic)
			cptobs[dic[x]],obs[dic[x]],vreal[dic[x]]=build_sc(x,train,test,classtr,cltest,dic,mest,dclas)
			cptnodes[dic[x]]=cpt(x,mest,dic,classtr)		

		"""

		#calcula de forma correcta las cpt_obs, haciendo validacion cruzada de 5 pliegues sobre en conjunto de entrenamiento
		train_co=train.copy()
		classtr_co=classtr.copy()
		fold_co=0
		nfolds_co=5
		rang_co=float(len(train))/float(nfolds_co)
		for fold_co in range(0,nfolds_co):
			tini_co=int(round(rang_co*fold_co))#instalar paquete en python
			tfin_co=int(round(rang_co*(fold_co+1)))

			test_co=np.zeros((tfin_co-tini_co,natt))
			cltest_co=np.zeros((tfin_co-tini_co,len(dic))).astype(int)

			#copia de train_co a test_co
			test_co[:,:]=train_co[:(tfin_co-tini_co),:]
			cltest_co[:,:]=classtr_co[:(tfin_co-tini_co),:]
			#remover elementos de train y classtr
			train_co=np.delete(train_co,np.s_[:(tfin_co-tini_co)],0)
			classtr_co=np.delete(classtr_co,np.s_[:(tfin_co-tini_co)],0)

			cola=[x for x in roots]
			visitados=[]		
			#for x in sorted(dic.keys()):
			while(len(cola)>0):
				x=cola.pop(0)

				p=[dic_inv[i] for i in range(0,len(mest)) if(mest[i,dic[x]]==1)]	# saco los padres de root
				band=True
				for z in p:
					if(not z in visitados):
						band=False
						break

				if(band):	#si todos los padres de "x" han sido visitados
					#agrego los hijos de "x" a la cola
					#	comprobando que no esten en la cola
					for i in range(0,len(mest)):
						#if(mest[dic[x],i]==1):
						if(mest[dic[x],i]==1):
							#obtengo el nombre del nodo
							#aux=[r[0] for r in dic.items() if(r[1]==i)]
							#agrego
							#if(len(aux)>0):
							if(not dic_inv[i] in cola and not dic_inv[i] in visitados):
								cola.append(dic_inv[i])

					#print("-------------------------------------------")
					#print("llego: "+x)
					#rec2(x,data,classes,mest,pathsc+"header_nm.arff",ftest,ctest,mest,dic)
					#build_classifiers(x,data,classes,mest,pathsc+"header_nm.arff",ftest,ctest,mest,dic)
					cptobs_aux[dic[x]],obs[dic[x]],vreal[dic[x]]=build_cpt_obs(x,train_co,test_co,classtr_co,cltest_co,dic,dic_inv,mest,dclas,dclas_prob,c_order,False, baseClassifier)
		
					#agrega a cptobs
					cptobs[dic[x]][0,0]=cptobs[dic[x]][0,0]+cptobs_aux[dic[x]][0,0]
					cptobs[dic[x]][0,1]=cptobs[dic[x]][0,1]+cptobs_aux[dic[x]][0,1]
					cptobs[dic[x]][1,0]=cptobs[dic[x]][1,0]+cptobs_aux[dic[x]][1,0]
					cptobs[dic[x]][1,1]=cptobs[dic[x]][1,1]+cptobs_aux[dic[x]][1,1]
					visitados.append(x)		#se agrega x a visitados
				else:
					cola.append(x)
			
			# concatenar test al final de train
			train_co=np.concatenate((train_co,test_co))
			classtr_co=np.concatenate((classtr_co,cltest_co))

			#se reinician las variables que se usaron
			#variable que guarda las probabilidades salidas de los clasificadores
			dclas_prob={}
			# guarda el orden de las clases en dclas_prob
			c_order=[0,0]
			#variable que guarda las salidas de los clasificadores
			dclas={}

		#suavizado laplaciano,
		for x in sorted(dic.keys()):
			cptobs[dic[x]][0,0]=cptobs[dic[x]][0,0]+1.0
			cptobs[dic[x]][0,1]=cptobs[dic[x]][0,1]+1.0
			cptobs[dic[x]][1,0]=cptobs[dic[x]][1,0]+1.0
			cptobs[dic[x]][1,1]=cptobs[dic[x]][1,1]+1.0

		#normalizacion 
		for x in sorted(dic.keys()):
			p00=cptobs[dic[x]][0,0]/(cptobs[dic[x]][0,0]+cptobs[dic[x]][0,1])
			p01=cptobs[dic[x]][0,1]/(cptobs[dic[x]][0,0]+cptobs[dic[x]][0,1])
			p10=cptobs[dic[x]][1,0]/(cptobs[dic[x]][1,0]+cptobs[dic[x]][1,1])
			p11=cptobs[dic[x]][1,1]/(cptobs[dic[x]][1,0]+cptobs[dic[x]][1,1])

			cptobs[dic[x]][0,0]=p00
			cptobs[dic[x]][0,1]=p01
			cptobs[dic[x]][1,0]=p10
			cptobs[dic[x]][1,1]=p11
		# fin calculo de fomar correcta cpt_obs ****************************************************************************

		#para cada nodo raiz construir su clasificador y luego para sus hijos
		#for x in roots:
		cola=[x for x in roots]
		visitados=[]		
		#for x in sorted(dic.keys()):
		while(len(cola)>0):
			x=cola.pop(0)

			p=[dic_inv[i] for i in range(0,len(mest)) if(mest[i,dic[x]]==1)]	# saco los padres de root
			band=True
			for z in p:
				if(not z in visitados):
					band=False
					break

			if(band):	#si todos los padres de "x" han sido visitados
				#agrego los hijos de "x" a la cola
				#	comprobando que no esten en la cola
				for i in range(0,len(mest)):
					#if(mest[dic[x],i]==1):
					if(mest[dic[x],i]==1):
						#obtengo el nombre del nodo
						#aux=[r[0] for r in dic.items() if(r[1]==i)]
						#agrego
						#if(len(aux)>0):
						if(not dic_inv[i] in cola and not dic_inv[i] in visitados):
							cola.append(dic_inv[i])

				#lib print("-------------------------------------------")
				#lib print("llego: "+x)
				#rec2(x,data,classes,mest,pathsc+"header_nm.arff",ftest,ctest,mest,dic)
				#build_classifiers(x,data,classes,mest,pathsc+"header_nm.arff",ftest,ctest,mest,dic)
				not_use,obs[dic[x]],vreal[dic[x]]=build_cpt_obs(x,train,test,classtr,cltest,dic,dic_inv,mest,dclas,dclas_prob,c_order,True, baseClassifier)		#build_sc(x,train,test,classtr,cltest,dic,mest,dclas,dclas_prob,c_order)
				#if(False):
				cptnodes[dic[x]]=cpt(x,mest,dic,classtr)
				visitados.append(x)		#se agrega x a visitados
			else:
				cola.append(x)


		#print("c_order")
		#print(c_order)
		#input("pausa*********************")
		########################3 construir la red bayesiana  #####################################33
		iaux=len(dic)
		dic_ob={}		#no esta siendo usada
		#for x in range(len(dic)):
		for x in dic.items():
			dic_ob[x[0]+"_ob"]=iaux
			iaux=iaux+1

		#llaves
		key_size={}	
		#llave main_root
		key_size["main_root"]=2
		for x in sorted(dic.keys()):
			key_size[x]=2
		#llaves de nodos observables
		for x in sorted(dic.keys()):
			key_size[x+"_ob"]=2

		#lib print("key_size:")
		#lib print(key_size)
		#values
		values=[]
		#factors
		factors=[]
		#diccionario de posiciones de los nodos en los arreglos de values/factors
		dicpos={}

		#agrego main_root a values, factors, dicpos
		dicpos["main_root"]=len(dicpos)
		factors.append(["main_root"])
		values.append(np.array([0.01,0.99]))

		#for x in roots:
		#	#add_factor(x,factors,mest)
		#	add_factor_values3(x,factors,values,cptnodes,mest,dic,dicpos)
		#esto se debe hacer con una cola, ya que deben estar asociados primero los padres de cada nodo ***************++++++++++++
		cola=[x for x in roots]
		visitados=[]		
		while(len(cola)>0):
			x=cola.pop(0)

			p=[dic_inv[i] for i in range(0,len(mest)) if(mest[i,dic[x]]==1)]	# saco los padres de root
			band=True
			for z in p:
				if(not z in visitados):
					band=False
					break

			if(band):	#si todos los padres de "x" han sido visitados
				#agrego los hijos de "x" a la cola
				#	comprobando que no esten en la cola
				for i in range(0,len(mest)):
					#if(mest[dic[x],i]==1):
					if(mest[dic[x],i]==1):
						#obtengo el nombre del nodo
						#aux=[r[0] for r in dic.items() if(r[1]==i)]
						#agrego
						#if(len(aux)>0):
						if(not dic_inv[i] in cola and not dic_inv[i] in visitados):
							cola.append(dic_inv[i])


				#add_factor(x,factors,mest)
				add_factor_values3(x,factors,values,cptnodes,mest,dic,dic_inv,dicpos)
				visitados.append(x)		#se agrega x a visitados
			else:
				cola.append(x)

		#values y factors de nodos observables
		for x in sorted(dic.keys()):
			factors.append([x,x+"_ob"])
			values.append(cptobs[dic[x]])
			dicpos[x+"_ob"]=len(dicpos)

		#lib print("diccionario de posiciones:")
		#lib print(dicpos)

		tree = jt.create_junction_tree(factors, key_size)
		#lib print("keys size: "+str(len(key_size)))
		#lib print("len factor: "+str(len(factors)))
		#lib print(factors)
		#lib print("len values: "+str(len(values)))
		#lib print(values)

		prop_values = tree.propagate(values)

		#lib print("prop values:")
		#lib print(prop_values)

		#lib print("********************************************+")
		#lib print("marginalizacion para cada nodo (no observable)")
		# Pr(sprinkler|wet_grass = 1)
		#marginal = np.sum(prop_values[dic["0/3/7"]], axis=0)
		for x in sorted(dic.keys()):
			marginal = np.sum(prop_values[dicpos[x]], axis=0)

			# The probabilities are unnormalized but we can calculate the normalized values:
			norm_marginal = marginal/np.sum(marginal)
			#print(x+": ")
			#print(norm_marginal)
		#lib print("********************************************+")


		################ Datos obersvados son pasados a la red #####################3	
		#necesito la posicion de cada nodo (normales y obasevables) en el vector de values/factors
		#quiza otro diccionario sea buena idea

		#este proceso debe ser hecho por cada ejemplo de test	

		#para cada ejemplo de test se modifica la red bayesiana
		outbn=np.zeros((len(cltest),len(dic))).astype(int)	#matriz que guarda la trayectoria consistente
		probm=np.zeros(len(dic))	#probabilidades marginales del ejemplo 
		contaux=0

	
		for i in range(len(cltest)):
			probm=np.zeros(len(dic))	#probabilidades marginales del ejemplo 
			#actualiza el tamaño de las variables observadas
			cond_sizes = key_size.copy()

			#hace algo similar con los valores
			cond_values = values.copy()
	
			if(type_prop=="just_positives"):
				# solo agregaran los que tienen observación positiva
				for j in sorted(dic.keys()):
					if(obs[dic[j]][i]==1):	#si es positivo se agrega como observación
						cond_sizes[x+"_ob"]=1

				#para cada prediccion obtenida
				#print("observaciones: ")
				#print("valor real real:")
				for j in sorted(dic.keys()):
					if(obs[dic[j]][i]==1):	#si es positivo eleimina la columna de negativos de la cpt
						cond_values[dicpos[j+"_ob"]]=cond_values[dicpos[j+"_ob"]][:,1:2]
					#else:					#si es negativo eleimina la columna de positivos de la cpt
					#	cond_values[dicpos[j+"_ob"]]=cond_values[dicpos[j+"_ob"]][:,0:1]
					#print(j+" ob: "+str(obs[dic[j]][i]))
					#print(j+": "+str(vreal[dic[j]][i] ))
					#outbn[i,dic[j]]=obs[dic[j]][i]		#no es una trayectoria consistente
			elif(type_prop=="all_probabilities"):
				for j in sorted(dic.keys()):
					cond_sizes[x+"_ob"]=1

				for j in sorted(dic.keys()):
					cond_values[dicpos[j+"_ob"]]=np.array([[ dclas_prob[j][i][c_order.index(0)]*cond_values[dicpos[j+"_ob"]][0][0]+ dclas_prob[j][i][c_order.index(1)]*cond_values[dicpos[j+"_ob"]][0][1] ],[ dclas_prob[j][i][c_order.index(0)]*cond_values[dicpos[j+"_ob"]][1][0]+ dclas_prob[j][i][c_order.index(1)]*cond_values[dicpos[j+"_ob"]][1][1] ]])

			else:
				print("ERROR!! you must choose 'just_positives' or 'all_probabilities' for belief propagation")
				exit()

			cond_tree = jt.create_junction_tree(factors, cond_sizes)

	

			#print("se propagara la info de los nodos ob")
			prop_values =  cond_tree.propagate(cond_values)
			#print(prop_values)

			#print("marginalizacion para cada nodo (no observable)")
			# Pr(sprinkler|wet_grass = 1)
			#marginal = np.sum(prop_values[dic["0/3/7"]], axis=0)
			for x in sorted(dic.keys()):
				marginal = np.sum(prop_values[dicpos[x]], axis=0)
				#print("mar ",marginal)

				for j in range(int(n_f[dic[x]])-1):		#segun el numero de padres es el numero de veces que se suma
					marginal = np.sum(marginal, axis=0)

				# The probabilities are unnormalized but we can calculate the normalized values:
				if(sum(marginal)!=0):
					norm_marginal = marginal/np.sum(marginal)
				else:
					norm_marginal = [0.0,0.0]
					#lib print("aun marginales en ceros :( ")

				#print(x+": ")
				#print(norm_marginal)
				if(not isinstance(norm_marginal,float)):	#deberia haber a los mas dos valores, el segundo valor corresponde al valor de true
					if(norm_marginal[1]==np.nan):
						probm[dic[x]]=0
						#lib print("hay probabilidad positiva igual a NaN")
					else:
						probm[dic[x]]=norm_marginal[1]
				else:
					#lib print("no me dio el vector marginal [-,+]")
					if(norm_marginal==np.nan):
						probm[dic[x]]=0
					else:
						probm[dic[x]]=norm_marginal

			#print("Prob marginales de un ejemplo:")
			#print(probm)

			#print("trayectoria mas probable")
			#selecciona la puntuacion para la trayectoria
			if(tscore=="GLB"):
				outbn[i]=score_glb(hojas,mest,probm,dic,n_f,we)
			elif(tscore=="SP"):			
				##print("trayectoria mas probable")
				##print(probm)
				#pmp=pathmostp(hojas,mest,probm,dic)
				##print(pmp)
	
				##se agrega la trayectoria mas probable
				#spl3=pmp.split("/")
				#spaux=spl3[0]
				#outbn[i,dic[spaux]]=1
				#for j in range(1,len(spl3)):
				#	spaux=spaux+"/"+spl3[j]
				#	outbn[i,dic[spaux]]=1
				outbn[i]=score_sp(dic_anc_leaves,dic_out_leaves,probm,dic)

			#print("Salida de un nodo:")
			#print(outbn[i])

			contaux=contaux+1
			#if(contaux>10):
			#	break
			#lib if((contaux%50)==0):
			#lib 	print("contaux: "+str(contaux))

		return outbn


