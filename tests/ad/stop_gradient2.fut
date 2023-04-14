-- Test chain rule.
def f (x: f64) = stop_gradient (f64.sin (x*x))

-- ==
-- entry: f_jvp
-- compiled input { 5.0 }
-- output { 0.0 }

entry f_jvp x =
  jvp f x 1

-- ==
-- entry: f_vjp
-- compiled input { 5.0 }
-- output { 0.0 }

entry f_vjp x =
  vjp f x 1
