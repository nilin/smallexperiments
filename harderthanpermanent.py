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


#	One=jnp.ones((n,n))	
#	MinusOne=jnp.concatenate([-jnp.ones((1,n)),jnp.ones((n-1,n))],axis=0)
#	
#	A=[jnp.array(n*[M])]
#	for i in range(1,n):
#		row=jnp.array(i*[MinusOne]+(n-i)*[One])
#		A.append(row)
#
#	A=jnp.stack(A,axis=0)
#	A=jnp.swapaxes(A,1,2)

def printtensor(A):
	n=A.shape[0]
	A_=jnp.reshape(A,(n*n,n*n))
	print(jnp.round(100*A_)/100)


#--------------------------------------------------------------------------------------------------------------
# tests
#--------------------------------------------------------------------------------------------------------------

	
def assertequal(a,b):
	print('\n'+str(a)+' =?= '+str(b)+'\n\n')
	assert(((a-b)**2)/(b**2)<.001)

	
def testpermanent():
	M=jnp.array([[1,1,1,1],[2,1,0,0],[3,0,1,0],[4,0,0,1]])
	print('testing permanent')
	assert(permanent(M)==10)


def verify_reduction(M):

	A=permanentreduction(M)

	print('Assert equality of')
	print('permanent of M,\nM=')
	print(M)
	print('\n\ngeneralized determinant of A,\nA=')
	printtensor(A)

	assertequal(permanent(M),gendet(A))



def test(ns):
	_,*keys=rnd.split(rnd.PRNGKey(0),1000)

	for n,key in zip(ns,keys):
		M=rnd.normal(key,(n,n))
		verify_reduction(M)


printtensor(inversiontensor(3))

if __name__=='__main__':
	test([2,3,4,5])



