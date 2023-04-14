-- The derivative of this should be `cos(x) * cos(x)`.
def f (x: f64) = (stop_gradient (f64.cos x)) * f64.sin x

-- ==
-- entry: f_jvp
-- compiled input { 0.0 }
-- output { 1.0 }

entry f_jvp x =
  jvp f x 1

-- ==
-- entry: f_vjp
-- compiled input { 0.0 }
-- output { 1.0 }

entry f_vjp x =
  vjp f x 1
