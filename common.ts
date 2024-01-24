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

type Materialize = (stuff: {
  x: number[];
  y: number[];
  theta: number[];
}) => number[][][];

const build = async (
  opts: Options,
): Promise<{
  render: Materialize;
  grad: lbfgs.Fn;
}> => {
  const fanout = fn([Real], Vec(opts.n, Real), (x) =>
    vec(opts.n, Real, () => x),
  );

  const sum = fn([Vec(opts.n, Real)], Real, (xs) => vjp(fanout)(0).grad(xs));

  const materialize = fn(
    [
      {
        x: Vec(opts.n, Real),
        y: Vec(opts.n, Real),
        theta: Vec(opts.n, Real),
      },
    ],
    Vec(opts.n, Vec(opts.m, Vec(2, Real))),
    ({ x, y, theta }) =>
      vec(opts.n, Vec(opts.m, Vec(2, Real)), (i) => {
        const points = [];
        for (let j = 0; j < opts.m; ++j) {
          const angle = add(theta[i], mul(j, div(2 * Math.PI, opts.m)));
          points.push([
            add(x[i], mul(opts.r, cos(angle))),
            add(y[i], mul(opts.r, sin(angle))),
          ]);
        }
        return points;
      }),
  );

  const lagrangian = fn(
    [
      {
        x: Vec(opts.n, Real),
        y: Vec(opts.n, Real),
        theta: Vec(opts.n, Real),
      },
    ],
    Real,
    ({ x, y, theta }) => {
      const polys = materialize({ x, y, theta });
      const canvas = sum(
        vec(opts.n, Real, (i) => {
          const points = polys[i];
          let total: Real = 0;
          for (let j = 0; j < opts.m; ++j) {
            const [x, y] = points[j];
            total = add(total, sqr(min(0, min(x, sub(size, x)))));
            total = add(total, sqr(min(0, min(y, sub(size, y)))));
          }
          return total;
        }),
      );
      const disjoint = sum(
        vec(opts.n, Real, (i) => {
          const poly1 = polys[i];
          const p: Real[][] = [];
          for (let k = 0; k < opts.m; ++k) {
            const [x, y] = poly1[k];
            p.push([x, y]);
          }
          return sum(
            vec(opts.n, Real, (j) => {
              const poly2 = polys[j];
              const q: Real[][] = [];
              for (let k = 0; k < opts.m; ++k) {
                const [x, y] = poly2[k];
                q.push([x, y]);
              }
              return select(
                ileq(opts.n, i, j),
                Real,
                0,
                sqr(max(0, neg(opts.minkowski(p, q)))),
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
          x: Vec(opts.n, Real),
          y: Vec(opts.n, Real),
          theta: Vec(opts.n, Real),
        },
      ],
      {
        z: Real,
        x: Vec(opts.n, Real),
        y: Vec(opts.n, Real),
        theta: Vec(opts.n, Real),
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
        x: xs.subarray(0, opts.n) as any,
        y: xs.subarray(opts.n, opts.n * 2) as any,
        theta: xs.subarray(opts.n * 2, opts.n * 3) as any,
      });

      // https://github.com/rose-lang/rose/issues/111
      dx.set(x as any, 0);
      dx.set(y as any, opts.n);
      dx.set(theta as any, opts.n * 2);

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
  opts: Options,
): { x: Float64Array; y: Float64Array; theta: Float64Array } => {
  const rng = seedrandom(opts.seed);
  const x = new Float64Array(opts.n);
  const y = new Float64Array(opts.n);
  const theta = new Float64Array(opts.n);
  for (let i = 0; i < opts.n; ++i) {
    x[i] = size * rng();
    y[i] = size * rng();
    theta[i] = 2 * Math.PI * rng();
  }
  return { x, y, theta };
};

const serialize = (opts: Options, { x, y, theta }: Polys): Float64Array => {
  const xs = new Float64Array(opts.n * 6);
  for (let i = 0; i < opts.n; ++i) {
    xs[i] = x[i];
    xs[i + opts.n] = y[i];
    xs[i + opts.n * 2] = theta[i];
  }
  return xs;
};

const deserialize = (opts: Options, xs: Float64Array): Polys => {
  const x = new Float64Array(opts.n);
  const y = new Float64Array(opts.n);
  const theta = new Float64Array(opts.n);
  for (let i = 0; i < opts.n; ++i) {
    x[i] = xs[i];
    y[i] = xs[i + opts.n];
    theta[i] = xs[i + opts.n * 2];
  }
  return { x, y, theta };
};

// https://stackoverflow.com/a/43122214
function bitCount(n: number) {
  n = n - ((n >> 1) & 0x55555555);
  n = (n & 0x33333333) + ((n >> 2) & 0x33333333);
  return (((n + (n >> 4)) & 0xf0f0f0f) * 0x1010101) >> 24;
}

const optimize = (opts: Options, f: lbfgs.Fn, polys: Polys): Polys => {
  const cfg: lbfgs.Config = {
    m: 17,
    armijo: 0.001,
    wolfe: 0.9,
    minInterval: 1e-9,
    maxSteps: 10,
    epsd: 1e-11,
  };
  const xs = serialize(opts, polys);
  const state = lbfgs.firstStep(cfg, f, xs);
  let fx: number;
  let i = 0;
  lbfgs.stepUntil(cfg, f, xs, state, (info) => {
    if (info.fx === fx) return true;
    fx = info.fx;
    if (bitCount(i) < 2) console.log(`i = ${i}, fx = ${fx}`);
    ++i;
  });
  return deserialize(opts, xs);
};

const svg = (
  opts: Options,
  materialize: Materialize,
  { x, y, theta }: Polys,
): string => {
  const hueFactor = 360 / opts.n;
  const polys = materialize({
    x: x as any,
    y: y as any,
    theta: theta as any,
  });
  const lines = [
    `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${size} ${size}">`,
  ];
  for (let i = 0; i < opts.n; ++i) {
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
  r: number;
  n: number;
  seed: string;
  minkowski: Minkowski;
  name: string;
}

export const run = async (opts: Options): Promise<number> => {
  const { render, grad: f } = await build(opts);
  const original = init(opts);
  const start = performance.now();
  const optimized = optimize(opts, f, original);
  const end = performance.now();
  const vector = svg(opts, render, optimized);
  const raster = new Resvg(vector, { fitTo: { mode: "width", value: 1000 } })
    .render()
    .asPng();
  await fs.mkdir(out, { recursive: true });
  await fs.writeFile(
    path.join(out, `${opts.name}-${opts.m}-${opts.n}.svg`),
    vector,
  );
  await fs.writeFile(
    path.join(out, `${opts.name}-${opts.m}-${opts.n}.png`),
    raster,
  );
  return end - start;
};
