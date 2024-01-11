import LinearAlgebra.norm
import Random
import Zygote

@kwdef struct Triangle
  a
  b
  c
end

function area(ab, bc, ca)
  0.25 * sqrt(
    (ab + bc + ca) * (-ab + bc + ca) * (ab - bc + ca) * (ab + bc - ca)
  )
end

function objective(triangles)
  sum(triangles) do (; a, b, c)
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

rate = 0.1

function optimize(triangles)
  for _ in 1:1000
    grad = Zygote.gradient(objective, triangles)
    triangles = map(zip(triangles, grad[1])) do (triangle, triangle_grad)
      Triangle(
        a=triangle.a - rate * triangle_grad.a,
        b=triangle.b - rate * triangle_grad.b,
        c=triangle.c - rate * triangle_grad.c,
      )
    end
  end
  triangles
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
