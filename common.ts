import { Resvg } from "@resvg/resvg-js";
import * as fs from "fs/promises";
import * as path from "path";
import {
  Bool,
  Dual,
  Nat,
  Real,
  Vec,
  add,
  and,
  compile,
  div,
  fn,
  geq,
  gt,
  iadd,
  ileq,
  ilt,
  imod,
  leq,
  lt,
  mul,
  neg,
  not,
  opaque,
  or,
  select,
  sqrt,
  sub,
  vec,
  vjp,
} from "rose";
import seedrandom from "seedrandom";
import * as lbfgs from "./lbfgs.js";

const sin = opaque([Real], Real, Math.sin);
const cos = opaque([Real], Real, Math.cos);

sin.jvp = fn([Dual], Dual, ({ re: x, du: dx }) => {
  const y = sin(x);
  const dy = mul(dx, cos(x));
  return { re: y, du: dy };
});

cos.jvp = fn([Dual], Dual, ({ re: x, du: dx }) => {
  const y = cos(x);
  const dy = mul(dx, neg(sin(x)));
  return { re: y, du: dy };
});

const sqr = (x: Real): Real => mul(x, x);

const max = (a: Real, b: Real) => select(gt(a, b), Real, a, b);

const min = (a: Real, b: Real): Real => select(lt(a, b), Real, a, b);

const clamp = (x: Real, l: Real, h: Real): Real =>
  select(lt(x, l), Real, l, select(lt(h, x), Real, h, x));

const all = (xs: Bool[]): Bool => xs.reduce((a, b) => and(a, b));

const vnot = (xs: Bool[]): Bool[] => xs.map((x) => not(x));

const vadd = (a: Vec<Real> | Real[], b: Vec<Real> | Real[]): Real[] => {
  return [add(a[0], b[0]), add(a[1], b[1])];
};

const vsub = (a: Vec<Real> | Real[], b: Vec<Real> | Real[]): Real[] => {
  return [sub(a[0], b[0]), sub(a[1], b[1])];
};

const vmul = (v: Real[], c: Real): Real[] => v.map((x) => mul(x, c));

const dot = (a: Real[], b: Real[]): Real =>
  a.map((x, i) => mul(x, b[i])).reduce((a, b) => add(a, b));

const cross = (a: Real[], b: Real[]): Real =>
  sub(mul(a[0], b[1]), mul(a[1], b[0]));

// https://iquilezles.org/articles/distfunctions2d/
const sdPolygon = (v: Real[][], p: Real[]): Real => {
  const N = v.length;
  let d = dot(vsub(p, v[0]), vsub(p, v[0]));
  let s: Real = 1.0;
  for (let i = 0, j = N - 1; i < N; j = i, i++) {
    const e = vsub(v[j], v[i]);
    const w = vsub(p, v[i]);
    const b = vsub(w, vmul(e, clamp(div(dot(w, e), dot(e, e)), 0.0, 1.0)));
    d = min(d, dot(b, b));
    const c = [
      geq(p[1], v[i][1]),
      lt(p[1], v[j][1]),
      gt(mul(e[0], w[1]), mul(e[1], w[0])),
    ];
    s = select(or(all(c), all(vnot(c))), Real, neg(s), s);
  }
  return mul(s, sqrt(d));
};

export type Minkowski = (p: Real[][], q: Real[][]) => Real;

export const exact = (m: number): Minkowski => {
  const reorder = fn([Vec(m, Vec(2, Real))], Vec(m + 2, Vec(2, Real)), (p) => {
    let i: Nat = 0;
    for (let j = 1; j < m; ++j) {
      const [xi, yi] = p[i] as any;
      const [xj, yj] = p[j];
      i = select(or(lt(yj, yi), and(leq(yj, yi), lt(xj, xi))), m, j, i);
    }
    const q: Vec<Real>[] = [];
    for (let j = 0; j < m + 2; ++j) {
      q.push(p[imod(m, iadd(m, i, j))]);
    }
    return q;
  });

  const minkowskiSum = (left: Real[][], right: Real[][]): Real[][] => {
    const p = reorder(left);
    const q = reorder(right);
    const r: Real[][] = [];
    let i: Nat = 0;
    let j: Nat = 0;
    for (let k = 0; k < 2 * m; ++k) {
      r.push(vadd(p[i], q[j]));
      const i1 = iadd(m + 2, i, 1);
      const j1 = iadd(m + 2, j, 1);
      const z = cross(vsub(p[i1], p[i]), vsub(q[j1], q[j]));
      ({ i, j } = select(
        and(ilt(m + 2, i, m), geq(z, 0)),
        { i: m + 2, j: m + 2 },
        { i: i1, j },
        { i, j: j1 },
      ));
    }
    return r;
  };

  return (p, q) =>
    sdPolygon(
      minkowskiSum(
        p,
        q.map(([x, y]) => [neg(x), neg(y)]),
      ),
      [0, 0],
    );
};

const size = 100;
const numPolys = 100;
const radius = 12 / Math.sqrt(3);

const fanout = fn([Real], Vec(numPolys, Real), (x) =>
  vec(numPolys, Real, () => x),
);

const sum = fn([Vec(numPolys, Real)], Real, (xs) => vjp(fanout)(0).grad(xs));

type Materialize = (stuff: {
  x: number[];
  y: number[];
  theta: number[];
}) => number[][][];

const build = async (
  m: number,
  minkowski: Minkowski,
): Promise<{
  render: Materialize;
  grad: lbfgs.Fn;
}> => {
  const materialize = fn(
    [
      {
        x: Vec(numPolys, Real),
        y: Vec(numPolys, Real),
        theta: Vec(numPolys, Real),
      },
    ],
    Vec(numPolys, Vec(m, Vec(2, Real))),
    ({ x, y, theta }) =>
      vec(numPolys, Vec(m, Vec(2, Real)), (i) => {
        const points = [];
        for (let j = 0; j < m; ++j) {
          const angle = add(theta[i], mul(j, div(2 * Math.PI, m)));
          points.push([
            add(x[i], mul(radius, cos(angle))),
            add(y[i], mul(radius, sin(angle))),
          ]);
        }
        return points;
      }),
  );

  const lagrangian = fn(
    [
      {
        x: Vec(numPolys, Real),
        y: Vec(numPolys, Real),
        theta: Vec(numPolys, Real),
      },
    ],
    Real,
    ({ x, y, theta }) => {
      const polys = materialize({ x, y, theta });
      const canvas = sum(
        vec(numPolys, Real, (i) => {
          const points = polys[i];
          let total: Real = 0;
          for (let j = 0; j < m; ++j) {
            const [x, y] = points[j];
            total = add(total, sqr(min(0, min(x, sub(size, x)))));
            total = add(total, sqr(min(0, min(y, sub(size, y)))));
          }
          return total;
        }),
      );
      const disjoint = sum(
        vec(numPolys, Real, (i) => {
          const poly1 = polys[i];
          const p: Real[][] = [];
          for (let k = 0; k < m; ++k) {
            const [x, y] = poly1[k];
            p.push([x, y]);
          }
          return sum(
            vec(numPolys, Real, (j) => {
              const poly2 = polys[j];
              const q: Real[][] = [];
              for (let k = 0; k < m; ++k) {
                const [x, y] = poly2[k];
                q.push([x, y]);
              }
              return select(
                ileq(numPolys, i, j),
                Real,
                0,
                sqr(max(0, neg(minkowski(p, q)))),
              );
            }),
          );
        }),
      );
      return add(canvas, disjoint);
    },
  );

  const compiled = await compile(
    fn(
      [
        {
          x: Vec(numPolys, Real),
          y: Vec(numPolys, Real),
          theta: Vec(numPolys, Real),
        },
      ],
      {
        z: Real,
        x: Vec(numPolys, Real),
        y: Vec(numPolys, Real),
        theta: Vec(numPolys, Real),
      },
      (inputs) => {
        const { ret, grad } = vjp(lagrangian)(inputs);
        const { x, y, theta } = grad(1);
        return { z: ret, x, y, theta };
      },
    ),
  );

  return {
    render: (await compile(materialize)) as any,
    grad: (xs: Float64Array, dx: Float64Array): number => {
      const { z, x, y, theta } = compiled({
        x: xs.subarray(0, numPolys) as any,
        y: xs.subarray(numPolys, numPolys * 2) as any,
        theta: xs.subarray(numPolys * 2, numPolys * 3) as any,
      });

      // https://github.com/rose-lang/rose/issues/111
      dx.set(x as any, 0);
      dx.set(y as any, numPolys);
      dx.set(theta as any, numPolys * 2);

      return z;
    },
  };
};

interface Polys {
  x: Float64Array;
  y: Float64Array;
  theta: Float64Array;
}

const init = (
  seed: string,
): { x: Float64Array; y: Float64Array; theta: Float64Array } => {
  const rng = seedrandom(seed);
  const x = new Float64Array(numPolys);
  const y = new Float64Array(numPolys);
  const theta = new Float64Array(numPolys);
  for (let i = 0; i < numPolys; ++i) {
    x[i] = size * rng();
    y[i] = size * rng();
    theta[i] = 2 * Math.PI * rng();
  }
  return { x, y, theta };
};

const serialize = ({ x, y, theta }: Polys): Float64Array => {
  const xs = new Float64Array(numPolys * 6);
  for (let i = 0; i < numPolys; ++i) {
    xs[i] = x[i];
    xs[i + numPolys] = y[i];
    xs[i + numPolys * 2] = theta[i];
  }
  return xs;
};

const deserialize = (xs: Float64Array): Polys => {
  const x = new Float64Array(numPolys);
  const y = new Float64Array(numPolys);
  const theta = new Float64Array(numPolys);
  for (let i = 0; i < numPolys; ++i) {
    x[i] = xs[i];
    y[i] = xs[i + numPolys];
    theta[i] = xs[i + numPolys * 2];
  }
  return { x, y, theta };
};

const optimize = (f: lbfgs.Fn, polys: Polys): Polys => {
  const cfg: lbfgs.Config = {
    m: 17,
    armijo: 0.001,
    wolfe: 0.9,
    minInterval: 1e-9,
    maxSteps: 10,
    epsd: 1e-11,
  };
  const xs = serialize(polys);
  const state = lbfgs.firstStep(cfg, f, xs);
  let fx: number;
  lbfgs.stepUntil(cfg, f, xs, state, (info) => {
    if (info.fx === fx) return true;
    console.log(info.fx);
    fx = info.fx;
  });
  return deserialize(xs);
};

const hueFactor = 360 / numPolys;

const svg = (materialize: Materialize, { x, y, theta }: Polys): string => {
  const polys = materialize({
    x: x as any,
    y: y as any,
    theta: theta as any,
  });
  const lines = [
    `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${size} ${size}">`,
  ];
  for (let i = 0; i < numPolys; ++i) {
    const poly = polys[i];
    const hue = Math.round(i * hueFactor);
    const points = poly.map(([x, y]) => `${x},${y}`).join(" ");
    lines.push(`  <polygon points="${points}" fill="hsl(${hue} 50% 50%)" />`);
  }
  lines.push("</svg>", "");
  return lines.join("\n");
};

const out = "out";

export interface Options {
  m: number;
  minkowski: Minkowski;
  name: string;
}

export const run = async (opts: Options): Promise<void> => {
  const { render, grad: f } = await build(opts.m, opts.minkowski);
  const original = init("");
  const optimized = optimize(f, original);
  const vector = svg(render, optimized);
  const raster = new Resvg(vector, { fitTo: { mode: "width", value: 1000 } })
    .render()
    .asPng();
  await fs.mkdir(out, { recursive: true });
  await fs.writeFile(path.join(out, `${opts.name}.svg`), vector);
  await fs.writeFile(path.join(out, `${opts.name}.png`), raster);
};
