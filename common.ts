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
  ieq,
  ileq,
  ilt,
  leq,
  lt,
  mul,
  neg,
  not,
  opaque,
  or,
  select,
  sqrt,
  struct,
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

const reorder = fn([Vec(3, Vec(2, Real))], Vec(5, Vec(2, Real)), (p) => {
  let i: Nat = 0;
  for (let j = 1; j < 3; ++j) {
    const [xi, yi] = p[i] as any;
    const [xj, yj] = p[j];
    i = select(or(lt(yj, yi), and(leq(yj, yi), lt(xj, xi))), 3, j, i);
  }
  const [a, b, c] = p;
  return select(
    ieq(3, i, 0),
    Vec(5, Vec(2, Real)),
    [a, b, c, a, b],
    select(
      ieq(3, i, 1),
      Vec(5, Vec(2, Real)),
      [b, c, a, b, c],
      [c, a, b, c, a],
    ),
  );
});

// note: only works on triangles
const minkowskiSum = (left: Real[][], right: Real[][]): Real[][] => {
  const p = reorder(left);
  const q = reorder(right);
  const r: Real[][] = [];
  let i: Nat = 0;
  let j: Nat = 0;
  for (let k = 0; k < 6; ++k) {
    r.push(vadd(p[i], q[j]));
    const i1 = iadd(5, i, 1);
    const j1 = iadd(5, j, 1);
    const z = cross(vsub(p[i1], p[i]), vsub(q[j1], q[j]));
    ({ i, j } = select(
      and(ilt(5, i, 3), geq(z, 0)),
      { i: 5, j: 5 },
      { i: i1, j },
      { i, j: j1 },
    ));
  }
  return r;
};

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

export const exact: Minkowski = (p, q) =>
  sdPolygon(
    minkowskiSum(
      p,
      q.map(([x, y]) => [neg(x), neg(y)]),
    ),
    [0, 0],
  );

const size = 100;
const numTriangles = 100;
const side = 12;

const fanout = fn([Real], Vec(numTriangles, Real), (x) =>
  vec(numTriangles, Real, () => x),
);

const sum = fn([Vec(numTriangles, Real)], Real, (xs) =>
  vjp(fanout)(0).grad(xs),
);

const Triangle = struct({
  ax: Real,
  ay: Real,
  bx: Real,
  by: Real,
  cx: Real,
  cy: Real,
});

const materialize = fn(
  [
    {
      x: Vec(numTriangles, Real),
      y: Vec(numTriangles, Real),
      theta: Vec(numTriangles, Real),
    },
  ],
  Vec(numTriangles, Triangle),
  ({ x, y, theta }) =>
    vec(numTriangles, Triangle, (i) => {
      const a = [x[i], y[i]];
      const b = vadd(a, [mul(side, cos(theta[i])), mul(side, sin(theta[i]))]);
      const c = vadd(a, [
        mul(side, cos(add(theta[i], Math.PI / 3))),
        mul(side, sin(add(theta[i], Math.PI / 3))),
      ]);
      return { ax: a[0], ay: a[1], bx: b[0], by: b[1], cx: c[0], cy: c[1] };
    }),
);

const build = async (minkowski: Minkowski): Promise<lbfgs.Fn> => {
  const lagrangian = fn(
    [
      {
        x: Vec(numTriangles, Real),
        y: Vec(numTriangles, Real),
        theta: Vec(numTriangles, Real),
      },
    ],
    Real,
    ({ x, y, theta }) => {
      const triangles = materialize({ x, y, theta });
      const canvas = sum(
        vec(numTriangles, Real, (i) => {
          const { ax, ay, bx, by, cx, cy } = triangles[i];
          return [
            [ax, ay],
            [bx, by],
            [cx, cy],
          ]
            .flatMap((v) => v.map((x) => sqr(min(0, min(x, sub(size, x))))))
            .reduce(add);
        }),
      );
      const disjoint = sum(
        vec(numTriangles, Real, (i) => {
          let { ax, ay, bx, by, cx, cy } = triangles[i];
          const a1 = [ax, ay];
          const b1 = [bx, by];
          const c1 = [cx, cy];
          const p = [a1, b1, c1];
          return sum(
            vec(numTriangles, Real, (j) => {
              ({ ax, ay, bx, by, cx, cy } = triangles[j]);
              const a2 = [ax, ay];
              const b2 = [bx, by];
              const c2 = [cx, cy];
              const q = [a2, b2, c2];
              return select(
                ileq(numTriangles, i, j),
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
          x: Vec(numTriangles, Real),
          y: Vec(numTriangles, Real),
          theta: Vec(numTriangles, Real),
        },
      ],
      {
        z: Real,
        x: Vec(numTriangles, Real),
        y: Vec(numTriangles, Real),
        theta: Vec(numTriangles, Real),
      },
      (inputs) => {
        const { ret, grad } = vjp(lagrangian)(inputs);
        const { x, y, theta } = grad(1);
        return { z: ret, x, y, theta };
      },
    ),
  );

  return (xs: Float64Array, dx: Float64Array): number => {
    const { z, x, y, theta } = compiled({
      x: xs.subarray(0, numTriangles) as any,
      y: xs.subarray(numTriangles, numTriangles * 2) as any,
      theta: xs.subarray(numTriangles * 2, numTriangles * 3) as any,
    });

    // https://github.com/rose-lang/rose/issues/111
    dx.set(x as any, 0);
    dx.set(y as any, numTriangles);
    dx.set(theta as any, numTriangles * 2);

    return z;
  };
};

interface Triangles {
  x: Float64Array;
  y: Float64Array;
  theta: Float64Array;
}

const init = (
  seed: string,
): { x: Float64Array; y: Float64Array; theta: Float64Array } => {
  const rng = seedrandom(seed);
  const x = new Float64Array(numTriangles);
  const y = new Float64Array(numTriangles);
  const theta = new Float64Array(numTriangles);
  for (let i = 0; i < numTriangles; ++i) {
    x[i] = size * rng();
    y[i] = size * rng();
    theta[i] = 2 * Math.PI * rng();
  }
  return { x, y, theta };
};

const serialize = ({ x, y, theta }: Triangles): Float64Array => {
  const xs = new Float64Array(numTriangles * 6);
  for (let i = 0; i < numTriangles; ++i) {
    xs[i] = x[i];
    xs[i + numTriangles] = y[i];
    xs[i + numTriangles * 2] = theta[i];
  }
  return xs;
};

const deserialize = (xs: Float64Array): Triangles => {
  const x = new Float64Array(numTriangles);
  const y = new Float64Array(numTriangles);
  const theta = new Float64Array(numTriangles);
  for (let i = 0; i < numTriangles; ++i) {
    x[i] = xs[i];
    y[i] = xs[i + numTriangles];
    theta[i] = xs[i + numTriangles * 2];
  }
  return { x, y, theta };
};

const optimize = (f: lbfgs.Fn, triangles: Triangles): Triangles => {
  const cfg: lbfgs.Config = {
    m: 17,
    armijo: 0.001,
    wolfe: 0.9,
    minInterval: 1e-9,
    maxSteps: 10,
    epsd: 1e-11,
  };
  const xs = serialize(triangles);
  const state = lbfgs.firstStep(cfg, f, xs);
  let fx: number;
  lbfgs.stepUntil(cfg, f, xs, state, (info) => {
    if (info.fx === fx) return true;
    console.log(info.fx);
    fx = info.fx;
  });
  return deserialize(xs);
};

const materializeFn = await compile(materialize);

const hueFactor = 360 / numTriangles;

const svg = ({ x, y, theta }: Triangles): string => {
  const triangles = materializeFn({
    x: x as any,
    y: y as any,
    theta: theta as any,
  });
  const lines = [
    `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${size} ${size}">`,
  ];
  for (let i = 0; i < numTriangles; ++i) {
    const { ax, ay, bx, by, cx, cy } = triangles[i];
    const hue = i * hueFactor;
    const points = `${ax},${ay} ${bx},${by} ${cx},${cy}`;
    lines.push(`  <polygon points="${points}" fill="hsl(${hue} 50% 50%)" />`);
  }
  lines.push("</svg>", "");
  return lines.join("\n");
};

export interface Options {
  minkowski: Minkowski;
  out: string;
}

export const run = async (opts: Options): Promise<void> => {
  const f = await build(opts.minkowski);
  const original = init("");
  const optimized = optimize(f, original);
  await fs.mkdir(path.dirname(opts.out), { recursive: true });
  await fs.writeFile(opts.out, svg(optimized));
};
