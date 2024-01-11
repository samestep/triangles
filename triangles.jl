module Lbfgs
# adapted from https://github.com/penrose/penrose/blob/0ab136a32e8e8d7df2dc896d5702b499a6b4b594/packages/core/src/engine/Lbfgs.ts

import LinearAlgebra.dot

@kwdef struct Config
  m
  armijo
  wolfe
  min_interval
  max_steps
  epsd
end

@kwdef struct SY
  s
  y
end

@kwdef struct State
  x
  grad
  s_y
end

function line_search!(cfg, f, x0, r, fx0, grad, x)
  duf_at_x0 = -dot(r, grad)

  a = 0
  b = Inf
  t = 1
  j = 0

  while true
    x .= x0 - t * r

    if abs(a - b) < cfg.min_interval || j > cfg.max_steps
      break
    end

    fx = f(x, grad)
    is_armijo = fx <= fx0 + cfg.armijo * t * duf_at_x0
    is_wolfe = -dot(r, grad) >= cfg.wolfe * duf_at_x0

    if !is_armijo
      b = t
    elseif !is_wolfe
      a = t
    else
      break
    end

    if b < Inf
      t = (a + b) / 2
    else
      t = 2 * a
    end

    j += 1
  end

  t
end

function first_step!(cfg, f, x)
  T = eltype(x)
  n = length(x)
  x0 = copy(x)

  grad = zeros(T, n)
  fx = f(x, grad)

  r = copy(grad)
  line_search!(cfg, f, x0, r, fx, grad, x)

  State(x=x0, grad=r, s_y=[])
end

@kwdef struct Info
  state
  fx
  r
  x
  t
end

function step_until!(cfg, f, x, state, stop)
  T = eltype(x)
  n = length(x)
  grad = zeros(T, n)

  rho = zeros(T, cfg.m)
  alpha = zeros(T, cfg.m)
  q = zeros(T, n)
  r = zeros(T, n)

  while true
    fx = f(x, grad)

    if length(state.s_y) < cfg.m
      push!(state.s_y, SY(s=x - state.x, y=grad - state.grad))
    else
      (; s, y) = last(state.s_y)
      s .= x - state.x
      y .= grad - state.grad
    end
    pushfirst!(state.s_y, pop!(state.s_y))

    state.x .= x
    state.grad .= grad

    for (j, pair) in enumerate(state.s_y)
      s_j = pair.s
      y_j = pair.y
      rho[j] = 1 / (dot(y_j, s_j) + cfg.epsd)
    end

    q .= grad

    for (j, pair) in enumerate(state.s_y)
      s_j = pair.s
      y_j = pair.y
      alpha_j = rho[j] * dot(s_j, q)
      alpha[j] = alpha_j
      q -= alpha_j * y_j
    end

    pair = first(state.s_y)
    s_k = pair.s
    y_k = pair.y
    gamma = dot(s_k, y_k) / (dot(y_k, y_k) + cfg.epsd)
    r .= gamma * q

    for (j, pair) in Iterators.reverse(enumerate(state.s_y))
      s_j = pair.s
      y_j = pair.y
      alpha_j = alpha[j]
      beta = rho[j] * dot(y_j, r)
      r += s_j * (alpha_j - beta)
    end

    t = line_search!(cfg, f, state.x, r, fx, grad, x)

    if stop(Info(state=state, fx=fx, r=r, x=x, t=t))
      break
    end
  end
end

end

import LinearAlgebra.norm
import Random
import Zygote

@kwdef struct Triangle
  a
  b
  c
end

chunks(serialized) = Iterators.partition(serialized, 6)

deserialize(chunk) = Triangle(
  a=chunk[1:2],
  b=chunk[3:4],
  c=chunk[5:6],
)

area(ab, bc, ca) =
  0.25 * sqrt(
    (ab + bc + ca) * (-ab + bc + ca) * (ab - bc + ca) * (ab + bc - ca)
  )

function objective(serialized)
  sum(chunks(serialized)) do chunk
    (; a, b, c) = deserialize(chunk)
    ab = norm(b - a)
    bc = norm(c - b)
    ca = norm(a - c)
    (area(ab, bc, ca) - 0.01)^2 + (ab - bc)^2 + (bc - ca)^2 + (ca - ab)^2
  end
end

function init(seed)
  Random.seed!(seed)
  triangles = []
  for _ = 1:10
    push!(triangles, Triangle(
      a=[rand(), rand()],
      b=[rand(), rand()],
      c=[rand(), rand()],
    ))
  end
  triangles
end

function obj_and_grad(x, dx)
  (; val, grad) = Zygote.withgradient(objective, x)
  dx .= first(grad)
  val
end

function optimize(triangles)
  cfg = Lbfgs.Config(
    m=17,
    armijo=0.001,
    wolfe=0.9,
    min_interval=1e-9,
    max_steps=10,
    epsd=1e-11,
  )
  x = vcat([vcat(t.a, t.b, t.c) for t in triangles]...)
  state = Lbfgs.first_step!(cfg, obj_and_grad, x)
  i = 0
  Lbfgs.step_until!(cfg, obj_and_grad, x, state, info -> (i += 1) >= 100)
  [deserialize(chunk) for chunk in chunks(x)]
end

vecstr(v) = join(v, ",")

function svg(triangles)
  lines = ["""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1 1">"""]
  for (i, (; a, b, c)) in enumerate(triangles)
    points = "$(vecstr(a)) $(vecstr(b)) $(vecstr(c))"
    fill = "hsl($(i * 360 / length(triangles)) 50% 50%)"
    push!(lines, """  <polygon points="$points" fill="$fill" />""")
  end
  push!(lines, "</svg>")
  push!(lines, "")
  join(lines, "\n")
end

function main()
  original = init(0)
  optimized = optimize(original)
  write("out.svg", svg(optimized))
end

main()
