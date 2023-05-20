-- Scan with 3x3 matrix multiplication.
-- ==
-- entry: fwd rev
-- compiled input { [[1f32,2f32,3f32,4f32,5f32,6f32,7f32,8f32,9f32],
-- [9f32,8f32,7f32,6f32,5f32,4f32,3f32,2f32,1f32],
-- [1f32,2f32,3f32,4f32,5f32,6f32,7f32,8f32,9f32],
-- [9f32,8f32,7f32,6f32,5f32,4f32,3f32,2f32,1f32]] }
-- output { [[[[1f32, 0f32, 0f32, 0f32, 0f32, 0f32, 0f32, 0f32, 0f32],
-- [0f32, 0f32, 0f32, 0f32, 0f32, 0f32, 0f32, 0f32, 0f32], [0f32,
-- 0f32, 0f32, 0f32, 0f32, 0f32, 0f32, 0f32, 0f32], [0f32, 0f32, 0f32,
-- 0f32, 0f32, 0f32, 0f32, 0f32, 0f32]], [[0f32, 1f32, 0f32, 0f32,
-- 0f32, 0f32, 0f32, 0f32, 0f32], [0f32, 0f32, 0f32, 0f32, 0f32, 0f32,
-- 0f32, 0f32, 0f32], [0f32, 0f32, 0f32, 0f32, 0f32, 0f32, 0f32, 0f32,
-- 0f32], [0f32, 0f32, 0f32, 0f32, 0f32, 0f32, 0f32, 0f32, 0f32]],
-- [[0f32, 0f32, 1f32, 0f32, 0f32, 0f32, 0f32, 0f32, 0f32], [0f32,
-- 0f32, 0f32, 0f32, 0f32, 0f32, 0f32, 0f32, 0f32], [0f32, 0f32, 0f32,
-- 0f32, 0f32, 0f32, 0f32, 0f32, 0f32], [0f32, 0f32, 0f32, 0f32, 0f32,
-- 0f32, 0f32, 0f32, 0f32]], [[0f32, 0f32, 0f32, 1f32, 0f32, 0f32,
-- 0f32, 0f32, 0f32], [0f32, 0f32, 0f32, 0f32, 0f32, 0f32, 0f32, 0f32,
-- 0f32], [0f32, 0f32, 0f32, 0f32, 0f32, 0f32, 0f32, 0f32, 0f32],
-- [0f32, 0f32, 0f32, 0f32, 0f32, 0f32, 0f32, 0f32, 0f32]], [[0f32,
-- 0f32, 0f32, 0f32, 1f32, 0f32, 0f32, 0f32, 0f32], [0f32, 0f32, 0f32,
-- 0f32, 0f32, 0f32, 0f32, 0f32, 0f32], [0f32, 0f32, 0f32, 0f32, 0f32,
-- 0f32, 0f32, 0f32, 0f32], [0f32, 0f32, 0f32, 0f32, 0f32, 0f32, 0f32,
-- 0f32, 0f32]], [[0f32, 0f32, 0f32, 0f32, 0f32, 1f32, 0f32, 0f32,
-- 0f32], [0f32, 0f32, 0f32, 0f32, 0f32, 0f32, 0f32, 0f32, 0f32],
-- [0f32, 0f32, 0f32, 0f32, 0f32, 0f32, 0f32, 0f32, 0f32], [0f32,
-- 0f32, 0f32, 0f32, 0f32, 0f32, 0f32, 0f32, 0f32]], [[0f32, 0f32,
-- 0f32, 0f32, 0f32, 0f32, 1f32, 0f32, 0f32], [0f32, 0f32, 0f32, 0f32,
-- 0f32, 0f32, 0f32, 0f32, 0f32], [0f32, 0f32, 0f32, 0f32, 0f32, 0f32,
-- 0f32, 0f32, 0f32], [0f32, 0f32, 0f32, 0f32, 0f32, 0f32, 0f32, 0f32,
-- 0f32]], [[0f32, 0f32, 0f32, 0f32, 0f32, 0f32, 0f32, 1f32, 0f32],
-- [0f32, 0f32, 0f32, 0f32, 0f32, 0f32, 0f32, 0f32, 0f32], [0f32,
-- 0f32, 0f32, 0f32, 0f32, 0f32, 0f32, 0f32, 0f32], [0f32, 0f32, 0f32,
-- 0f32, 0f32, 0f32, 0f32, 0f32, 0f32]], [[0f32, 0f32, 0f32, 0f32,
-- 0f32, 0f32, 0f32, 0f32, 1f32], [0f32, 0f32, 0f32, 0f32, 0f32, 0f32,
-- 0f32, 0f32, 0f32], [0f32, 0f32, 0f32, 0f32, 0f32, 0f32, 0f32, 0f32,
-- 0f32], [0f32, 0f32, 0f32, 0f32, 0f32, 0f32, 0f32, 0f32, 0f32]]],
-- [[[9f32, 6f32, 3f32, 0f32, 0f32, 0f32, 0f32, 0f32, 0f32], [1f32,
-- 0f32, 0f32, 2f32, 0f32, 0f32, 3f32, 0f32, 0f32], [0f32, 0f32, 0f32,
-- 0f32, 0f32, 0f32, 0f32, 0f32, 0f32], [0f32, 0f32, 0f32, 0f32, 0f32,
-- 0f32, 0f32, 0f32, 0f32]], [[8f32, 5f32, 2f32, 0f32, 0f32, 0f32,
-- 0f32, 0f32, 0f32], [0f32, 1f32, 0f32, 0f32, 2f32, 0f32, 0f32, 3f32,
-- 0f32], [0f32, 0f32, 0f32, 0f32, 0f32, 0f32, 0f32, 0f32, 0f32],
-- [0f32, 0f32, 0f32, 0f32, 0f32, 0f32, 0f32, 0f32, 0f32]], [[7f32,
-- 4f32, 1f32, 0f32, 0f32, 0f32, 0f32, 0f32, 0f32], [0f32, 0f32, 1f32,
-- 0f32, 0f32, 2f32, 0f32, 0f32, 3f32], [0f32, 0f32, 0f32, 0f32, 0f32,
-- 0f32, 0f32, 0f32, 0f32], [0f32, 0f32, 0f32, 0f32, 0f32, 0f32, 0f32,
-- 0f32, 0f32]], [[0f32, 0f32, 0f32, 9f32, 6f32, 3f32, 0f32, 0f32,
-- 0f32], [4f32, 0f32, 0f32, 5f32, 0f32, 0f32, 6f32, 0f32, 0f32],
-- [0f32, 0f32, 0f32, 0f32, 0f32, 0f32, 0f32, 0f32, 0f32], [0f32,
-- 0f32, 0f32, 0f32, 0f32, 0f32, 0f32, 0f32, 0f32]], [[0f32, 0f32,
-- 0f32, 8f32, 5f32, 2f32, 0f32, 0f32, 0f32], [0f32, 4f32, 0f32, 0f32,
-- 5f32, 0f32, 0f32, 6f32, 0f32], [0f32, 0f32, 0f32, 0f32, 0f32, 0f32,
-- 0f32, 0f32, 0f32], [0f32, 0f32, 0f32, 0f32, 0f32, 0f32, 0f32, 0f32,
-- 0f32]], [[0f32, 0f32, 0f32, 7f32, 4f32, 1f32, 0f32, 0f32, 0f32],
-- [0f32, 0f32, 4f32, 0f32, 0f32, 5f32, 0f32, 0f32, 6f32], [0f32,
-- 0f32, 0f32, 0f32, 0f32, 0f32, 0f32, 0f32, 0f32], [0f32, 0f32, 0f32,
-- 0f32, 0f32, 0f32, 0f32, 0f32, 0f32]], [[0f32, 0f32, 0f32, 0f32,
-- 0f32, 0f32, 9f32, 6f32, 3f32], [7f32, 0f32, 0f32, 8f32, 0f32, 0f32,
-- 9f32, 0f32, 0f32], [0f32, 0f32, 0f32, 0f32, 0f32, 0f32, 0f32, 0f32,
-- 0f32], [0f32, 0f32, 0f32, 0f32, 0f32, 0f32, 0f32, 0f32, 0f32]],
-- [[0f32, 0f32, 0f32, 0f32, 0f32, 0f32, 8f32, 5f32, 2f32], [0f32,
-- 7f32, 0f32, 0f32, 8f32, 0f32, 0f32, 9f32, 0f32], [0f32, 0f32, 0f32,
-- 0f32, 0f32, 0f32, 0f32, 0f32, 0f32], [0f32, 0f32, 0f32, 0f32, 0f32,
-- 0f32, 0f32, 0f32, 0f32]], [[0f32, 0f32, 0f32, 0f32, 0f32, 0f32,
-- 7f32, 4f32, 1f32], [0f32, 0f32, 7f32, 0f32, 0f32, 8f32, 0f32, 0f32,
-- 9f32], [0f32, 0f32, 0f32, 0f32, 0f32, 0f32, 0f32, 0f32, 0f32],
-- [0f32, 0f32, 0f32, 0f32, 0f32, 0f32, 0f32, 0f32, 0f32]]], [[[90f32,
-- 54f32, 18f32, 0f32, 0f32, 0f32, 0f32, 0f32, 0f32], [1f32, 4f32,
-- 7f32, 2f32, 8f32, 14f32, 3f32, 12f32, 21f32], [30f32, 0f32, 0f32,
-- 24f32, 0f32, 0f32, 18f32, 0f32, 0f32], [0f32, 0f32, 0f32, 0f32,
-- 0f32, 0f32, 0f32, 0f32, 0f32]], [[114f32, 69f32, 24f32, 0f32, 0f32,
-- 0f32, 0f32, 0f32, 0f32], [2f32, 5f32, 8f32, 4f32, 10f32, 16f32,
-- 6f32, 15f32, 24f32], [0f32, 30f32, 0f32, 0f32, 24f32, 0f32, 0f32,
-- 18f32, 0f32], [0f32, 0f32, 0f32, 0f32, 0f32, 0f32, 0f32, 0f32,
-- 0f32]], [[138f32, 84f32, 30f32, 0f32, 0f32, 0f32, 0f32, 0f32,
-- 0f32], [3f32, 6f32, 9f32, 6f32, 12f32, 18f32, 9f32, 18f32, 27f32],
-- [0f32, 0f32, 30f32, 0f32, 0f32, 24f32, 0f32, 0f32, 18f32], [0f32,
-- 0f32, 0f32, 0f32, 0f32, 0f32, 0f32, 0f32, 0f32]], [[0f32, 0f32,
-- 0f32, 90f32, 54f32, 18f32, 0f32, 0f32, 0f32], [4f32, 16f32, 28f32,
-- 5f32, 20f32, 35f32, 6f32, 24f32, 42f32], [84f32, 0f32, 0f32, 69f32,
-- 0f32, 0f32, 54f32, 0f32, 0f32], [0f32, 0f32, 0f32, 0f32, 0f32,
-- 0f32, 0f32, 0f32, 0f32]], [[0f32, 0f32, 0f32, 114f32, 69f32, 24f32,
-- 0f32, 0f32, 0f32], [8f32, 20f32, 32f32, 10f32, 25f32, 40f32, 12f32,
-- 30f32, 48f32], [0f32, 84f32, 0f32, 0f32, 69f32, 0f32, 0f32, 54f32,
-- 0f32], [0f32, 0f32, 0f32, 0f32, 0f32, 0f32, 0f32, 0f32, 0f32]],
-- [[0f32, 0f32, 0f32, 138f32, 84f32, 30f32, 0f32, 0f32, 0f32],
-- [12f32, 24f32, 36f32, 15f32, 30f32, 45f32, 18f32, 36f32, 54f32],
-- [0f32, 0f32, 84f32, 0f32, 0f32, 69f32, 0f32, 0f32, 54f32], [0f32,
-- 0f32, 0f32, 0f32, 0f32, 0f32, 0f32, 0f32, 0f32]], [[0f32, 0f32,
-- 0f32, 0f32, 0f32, 0f32, 90f32, 54f32, 18f32], [7f32, 28f32, 49f32,
-- 8f32, 32f32, 56f32, 9f32, 36f32, 63f32], [138f32, 0f32, 0f32,
-- 114f32, 0f32, 0f32, 90f32, 0f32, 0f32], [0f32, 0f32, 0f32, 0f32,
-- 0f32, 0f32, 0f32, 0f32, 0f32]], [[0f32, 0f32, 0f32, 0f32, 0f32,
-- 0f32, 114f32, 69f32, 24f32], [14f32, 35f32, 56f32, 16f32, 40f32,
-- 64f32, 18f32, 45f32, 72f32], [0f32, 138f32, 0f32, 0f32, 114f32,
-- 0f32, 0f32, 90f32, 0f32], [0f32, 0f32, 0f32, 0f32, 0f32, 0f32,
-- 0f32, 0f32, 0f32]], [[0f32, 0f32, 0f32, 0f32, 0f32, 0f32, 138f32,
-- 84f32, 30f32], [21f32, 42f32, 63f32, 24f32, 48f32, 72f32, 27f32,
-- 54f32, 81f32], [0f32, 0f32, 138f32, 0f32, 0f32, 114f32, 0f32, 0f32,
-- 90f32], [0f32, 0f32, 0f32, 0f32, 0f32, 0f32, 0f32, 0f32, 0f32]]],
-- [[[1908f32, 1152f32, 396f32, 0f32, 0f32, 0f32, 0f32, 0f32, 0f32],
-- [30f32, 84f32, 138f32, 60f32, 168f32, 276f32, 90f32, 252f32,
-- 414f32], [270f32, 180f32, 90f32, 216f32, 144f32, 72f32, 162f32,
-- 108f32, 54f32], [252f32, 0f32, 0f32, 324f32, 0f32, 0f32, 396f32,
-- 0f32, 0f32]], [[1566f32, 945f32, 324f32, 0f32, 0f32, 0f32, 0f32,
-- 0f32, 0f32], [24f32, 69f32, 114f32, 48f32, 138f32, 228f32, 72f32,
-- 207f32, 342f32], [240f32, 150f32, 60f32, 192f32, 120f32, 48f32,
-- 144f32, 90f32, 36f32], [0f32, 252f32, 0f32, 0f32, 324f32, 0f32,
-- 0f32, 396f32, 0f32]], [[1224f32, 738f32, 252f32, 0f32, 0f32, 0f32,
-- 0f32, 0f32, 0f32], [18f32, 54f32, 90f32, 36f32, 108f32, 180f32,
-- 54f32, 162f32, 270f32], [210f32, 120f32, 30f32, 168f32, 96f32,
-- 24f32, 126f32, 72f32, 18f32], [0f32, 0f32, 252f32, 0f32, 0f32,
-- 324f32, 0f32, 0f32, 396f32]], [[0f32, 0f32, 0f32, 1908f32, 1152f32,
-- 396f32, 0f32, 0f32, 0f32], [120f32, 336f32, 552f32, 150f32, 420f32,
-- 690f32, 180f32, 504f32, 828f32], [756f32, 504f32, 252f32, 621f32,
-- 414f32, 207f32, 486f32, 324f32, 162f32], [738f32, 0f32, 0f32,
-- 945f32, 0f32, 0f32, 1152f32, 0f32, 0f32]], [[0f32, 0f32, 0f32,
-- 1566f32, 945f32, 324f32, 0f32, 0f32, 0f32], [96f32, 276f32, 456f32,
-- 120f32, 345f32, 570f32, 144f32, 414f32, 684f32], [672f32, 420f32,
-- 168f32, 552f32, 345f32, 138f32, 432f32, 270f32, 108f32], [0f32,
-- 738f32, 0f32, 0f32, 945f32, 0f32, 0f32, 1152f32, 0f32]], [[0f32,
-- 0f32, 0f32, 1224f32, 738f32, 252f32, 0f32, 0f32, 0f32], [72f32,
-- 216f32, 360f32, 90f32, 270f32, 450f32, 108f32, 324f32, 540f32],
-- [588f32, 336f32, 84f32, 483f32, 276f32, 69f32, 378f32, 216f32,
-- 54f32], [0f32, 0f32, 738f32, 0f32, 0f32, 945f32, 0f32, 0f32,
-- 1152f32]], [[0f32, 0f32, 0f32, 0f32, 0f32, 0f32, 1908f32, 1152f32,
-- 396f32], [210f32, 588f32, 966f32, 240f32, 672f32, 1104f32, 270f32,
-- 756f32, 1242f32], [1242f32, 828f32, 414f32, 1026f32, 684f32,
-- 342f32, 810f32, 540f32, 270f32], [1224f32, 0f32, 0f32, 1566f32,
-- 0f32, 0f32, 1908f32, 0f32, 0f32]], [[0f32, 0f32, 0f32, 0f32, 0f32,
-- 0f32, 1566f32, 945f32, 324f32], [168f32, 483f32, 798f32, 192f32,
-- 552f32, 912f32, 216f32, 621f32, 1026f32], [1104f32, 690f32, 276f32,
-- 912f32, 570f32, 228f32, 720f32, 450f32, 180f32], [0f32, 1224f32,
-- 0f32, 0f32, 1566f32, 0f32, 0f32, 1908f32, 0f32]], [[0f32, 0f32,
-- 0f32, 0f32, 0f32, 0f32, 1224f32, 738f32, 252f32], [126f32, 378f32,
-- 630f32, 144f32, 432f32, 720f32, 162f32, 486f32, 810f32], [966f32,
-- 552f32, 138f32, 798f32, 456f32, 114f32, 630f32, 360f32, 90f32],
-- [0f32, 0f32, 1224f32, 0f32, 0f32, 1566f32, 0f32, 0f32, 1908f32]]]]
-- }

def mm3by3  (a1: f32, b1: f32, c1: f32, d1: f32, e1: f32, f1: f32, g1: f32, h1: f32, i1: f32)
            (a2: f32, b2: f32, c2: f32, d2: f32, e2: f32, f2: f32, g2: f32, h2: f32, i2: f32) =
  ( a1*a2 + b1*d2 + c1*g2
  , a1*b2 + b1*e2 + c1*h2
  , a1*c2 + b1*f2 + c1*i2

  , d1*a2 + e1*d2 + f1*g2
  , d1*b2 + e1*e2 + f1*h2
  , d1*c2 + e1*f2 + f1*i2

  , g1*a2 + h1*d2 + i1*g2
  , g1*b2 + h1*e2 + i1*h2
  , g1*c2 + h1*f2 + i1*i2
  )

def primal3 [n] (xs: [n](f32,f32,f32,f32,f32,f32,f32,f32,f32)) =
  scan mm3by3 (1,0,0, 0,1,0, 0,0,1) xs

def fromarrs3 = map (\(x: [9]f32) -> (x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8]))
def toarrs3 = map (\(a,b,c,d,e,f,g,h,i) -> [a,b,c,d,e,f,g,h,i])

def onehot_2d n m x y =
  tabulate_2d n m (\i j -> f32.bool((i,j) == (x,y)))

entry fwd [n] (input: [n][9]f32) : [n][9][n][9]f32 =
  let input = fromarrs3 input
  in tabulate (n*9) (\i -> jvp primal3 input (fromarrs3 (onehot_2d n 9 (i/9) (i%9))))
     |> map toarrs3 |> transpose |> map transpose |> map (map unflatten)

entry rev [n] (input: [n][9]f32) : [n][9][n][9]f32 =
  let input = fromarrs3 input
  in tabulate (n*9) (\i -> vjp primal3 input (fromarrs3 (onehot_2d n 9 (i/9) (i%9))))
     |> unflatten |> map (map toarrs3)
