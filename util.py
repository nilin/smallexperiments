import numpy as np
import math
import pickle
import time
import copy
import jax
import jax.numpy as jnp




# apply matrix A[...,:,:] on X[...,:,.]
@jax.jit
def apply_on_n(A,X):

	_=jnp.dot(A,X)
	out= jnp.swapaxes(_,len(A.shape)-2,-2)

	return out


@jax.jit
def flatten_first(X):
	blocksize=X.shape[0]*X.shape[1]
	shape=X.shape[2:]
	return jnp.reshape(X,(blocksize,)+shape)
	


# for universality.train # ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ 



@jax.jit
def allmatrixproducts(As,Bs):
	products=apply_on_n(As,Bs)
	return flatten_first(products)


