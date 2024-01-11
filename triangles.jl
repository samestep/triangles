#!/usr/bin/env julia

import Base.@kwdef
import Random

@kwdef struct Triangle
  a
  b
  c
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
  optimized = original
  write("out.svg", svg(optimized))
end

main()
