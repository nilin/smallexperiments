# nilin

import jax.numpy as jnp
import permutations_simple as PS
import jax.random as rnd
import math



def perms(n):
	ps,signs=PS.allpermtuples(n)
	return [p for p in ps],list(signs)



def gendet(A):
	n=A.shape[0]
	out=0
	for p,sign in zip(*perms(n)):
		prod=1
		for i in range(n):
			for j in range(n):
				prod=prod*A[i,j,p[i],p[j]]
		out=out+sign*prod
	return out


def permanent(M):
	n=M.shape[0]
	ps,_=perms(n)
	out=0
	for p in ps:
		prod=1
		for i in range(n):
			prod=prod*M[i,p[i]]
		out=out+prod
	return out
				

def detreduction(M):
	n=M.shape[0]
	M_=jnp.reshape(M,(1,n,1,n))
	top=jnp.repeat(M_,n,axis=-2)
	bottom=jnp.ones((n-1,n,n,n))
	return jnp.concatenate([top,bottom],axis=0)


def inversiontensor(n):
	i_before_j=jnp.triu(jnp.ones((n,n)))
	pi_after_pj=1-i_before_j
	inversion01=i_before_j[:,:,None,None]*pi_after_pj[None,None,:,:]
	return 1-2*inversion01


def permanentreduction(M):
	n=M.shape[0]
	return detreduction(M)*inversiontensor(n)



def printtensor(A):
	n=A.shape[0]
	A_=jnp.reshape(A,(n*n,n*n))
	print(jnp.round(100*A_)/100)


#--------------------------------------------------------------------------------------------------------------
# tests
#--------------------------------------------------------------------------------------------------------------

	
def assertequal(a,b):
	print('\n'+str(a)+' =?= '+str(b))
	assert(((a-b)**2)/(b**2)<.001)

	
def testpermanent():
	M=jnp.array([[1,1,1,1],[2,1,0,0],[3,0,1,0],[4,0,0,1]])
	print('testing permanent')
	assert(permanent(M)==10)


def verify_reduction(M):

	A=permanentreduction(M)
	print('Assert permanent(M)==gendet(A(M)).')
	assertequal(permanent(M),gendet(A))
	print('test passed'+100*'-'+'\n\n')



def test(ns):
	_,*keys=rnd.split(rnd.PRNGKey(0),1000)

	for n,key in zip(ns,keys):
		print('n='+str(n))
		M=rnd.normal(key,(n,n))
		verify_reduction(M)



def displayreduction(n):
	print('Example of reduction:')
	M=rnd.normal(rnd.PRNGKey(0),(n,n))
	print('M=')
	print(M)
	print('\nA(M)=')
	printtensor(permanentreduction(M))



if __name__=='__main__':

	print('\n\n\nGiven M we construct A(M) such that permanent(M)=gendet(A(M)).')
	print('This shows that the generalized determinant is harder than the permanent.\n')
	displayreduction(3)

	input('\npress enter to test')
	test([1,2,3,4,5])




