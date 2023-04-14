-- Defining the gradient of a non-differentiable function.
-- From the JAX straight-through estimator example:
-- https://jax.readthedocs.io/en/latest/jax-101/04-advanced-autodiff.html#straight-through-estimator-using-stop-gradient

def f (x: f64) = f64.round x

def straight_through_f (x: f64) =
  let zero = x - stop_gradient x
  in zero + (stop_gradient (f x))

-- ==
-- entry: f_jvp
-- compiled input { 3.2 }
-- output { 1.0 }

entry f_jvp x =
   jvp straight_through_f x 1

-- ==
-- entry: f_vjp
-- compiled input { 3.2 }
-- output { 1.0 }

entry f_vjp x =
   vjp straight_through_f x 1
