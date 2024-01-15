#!/usr/bin/env bun

import {
  Bool,
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
  or,
  select,
  sqrt,
  sub,
  vec,
  vjp,
} from "rose";
import seedrandom from "seedrandom";
import * as optimizer from "./optimizer.js";

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

const norm = (v: Real[]): Real => sqrt(dot(v, v));

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
const minkowskiSum = (
  left: Vec<Vec<Real>>,
  right: Vec<Vec<Real>>,
): Real[][] => {
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

type Vec2 = number[];

interface Triangle {
  a: Vec2;
  b: Vec2;
  c: Vec2;
}

const area = (ab: Real, bc: Real, ca: Real): Real => {
  const one = add(add(ab, bc), ca);
  const two = add(add(neg(ab), bc), ca);
  const three = add(sub(ab, bc), ca);
  const four = sub(add(ab, bc), ca);

  return mul(0.25, sqrt(mul(mul(mul(one, two), three), four)));
};

const clockwise = (a: Real[], b: Real[], c: Real[]): Bool =>
  lt(
    mul(sub(b[0], a[0]), sub(c[1], a[1])),
    mul(sub(c[0], a[0]), sub(b[1], a[1])),
  );

const numTriangles = 10;

const fanout = fn([Real], Vec(numTriangles, Real), (x) =>
  vec(numTriangles, Real, () => x),
);

const sum = fn([Vec(numTriangles, Real)], Real, (xs) =>
  vjp(fanout)(0).grad(xs),
);

const size = 100;

const lagrangian = fn(
  [
    {
      ax: Vec(numTriangles, Real),
      ay: Vec(numTriangles, Real),
      bx: Vec(numTriangles, Real),
      by: Vec(numTriangles, Real),
      cx: Vec(numTriangles, Real),
      cy: Vec(numTriangles, Real),
      weight: Real,
    },
  ],
  Real,
  ({ ax, ay, bx, by, cx, cy, weight }) => {
    const equilateral = sum(
      vec(numTriangles, Real, (i) => {
        const a = [ax[i], ay[i]];
        const b = [bx[i], by[i]];
        const c = [cx[i], cy[i]];

        const ab = norm(vsub(b, a));
        const bc = norm(vsub(c, b));
        const ca = norm(vsub(a, c));

        return add(
          sqr(sub(area(ab, bc, ca), (size / 10) ** 2)),
          add(add(sqr(sub(ab, bc)), sqr(sub(bc, ca))), sqr(sub(ca, ab))),
        );
      }),
    );
    const canvas = sum(
      vec(numTriangles, Real, (i) => {
        return [
          [ax[i], ay[i]],
          [bx[i], by[i]],
          [cx[i], cy[i]],
        ]
          .flatMap((v) => v.map((x) => sqr(min(0, min(x, sub(size, x))))))
          .reduce(add);
      }),
    );
    const disjoint = sum(
      vec(numTriangles, Real, (i) => {
        const a1 = [ax[i], ay[i]];
        const b1 = [bx[i], by[i]];
        const c1 = [cx[i], cy[i]];
        const p = select(
          clockwise(a1, b1, c1),
          Vec(3, Vec(2, Real)),
          [c1, b1, a1],
          [a1, b1, c1],
        );
        return sum(
          vec(numTriangles, Real, (j) => {
            const a2 = [neg(ax[j]), neg(ay[j])];
            const b2 = [neg(bx[j]), neg(by[j])];
            const c2 = [neg(cx[j]), neg(cy[j])];
            const q = select(
              clockwise(a1, b1, c1),
              Vec(3, Vec(2, Real)),
              [c2, b2, a2],
              [a2, b2, c2],
            );
            return select(
              ileq(numTriangles, i, j),
              Real,
              0,
              sqr(max(0, neg(sdPolygon(minkowskiSum(p, q), [0, 0])))),
            );
          }),
        );
      }),
    );
    return add(equilateral, mul(weight, add(canvas, disjoint)));
  },
);

const compiled = await compile(
  fn(
    [
      {
        ax: Vec(numTriangles, Real),
        ay: Vec(numTriangles, Real),
        bx: Vec(numTriangles, Real),
        by: Vec(numTriangles, Real),
        cx: Vec(numTriangles, Real),
        cy: Vec(numTriangles, Real),
        weight: Real,
      },
    ],
    {
      z: Real,
      ax: Vec(numTriangles, Real),
      ay: Vec(numTriangles, Real),
      bx: Vec(numTriangles, Real),
      by: Vec(numTriangles, Real),
      cx: Vec(numTriangles, Real),
      cy: Vec(numTriangles, Real),
    },
    (inputs) => {
      const { ret, grad } = vjp(lagrangian)(inputs);
      const { ax, ay, bx, by, cx, cy } = grad(1);
      return { z: ret, ax, ay, bx, by, cx, cy };
    },
  ),
);

const grad: optimizer.Fn = (
  x: Float64Array,
  weight: number,
  dx: Float64Array,
): number => {
  const { z, ax, ay, bx, by, cx, cy } = compiled({
    ax: x.subarray(0, numTriangles) as any,
    ay: x.subarray(numTriangles, numTriangles * 2) as any,
    bx: x.subarray(numTriangles * 2, numTriangles * 3) as any,
    by: x.subarray(numTriangles * 3, numTriangles * 4) as any,
    cx: x.subarray(numTriangles * 4, numTriangles * 5) as any,
    cy: x.subarray(numTriangles * 5, numTriangles * 6) as any,
    weight,
  });

  // https://github.com/rose-lang/rose/issues/111
  dx.set(ax as any, 0);
  dx.set(ay as any, numTriangles);
  dx.set(bx as any, numTriangles * 2);
  dx.set(by as any, numTriangles * 3);
  dx.set(cx as any, numTriangles * 4);
  dx.set(cy as any, numTriangles * 5);

  return z;
};

const init = (seed: string): Triangle[] => {
  const rng = seedrandom(seed);
  const triangles: Triangle[] = [];
  for (let i = 0; i < numTriangles; ++i) {
    const a = [rng() * size, rng() * size];
    const b = [rng() * size, rng() * size];
    const c = [rng() * size, rng() * size];
    triangles.push({ a, b, c });
  }
  return triangles;
};

const serialize = (triangles: Triangle[]): Float64Array => {
  const x = new Float64Array(numTriangles * 6);
  for (let i = 0; i < numTriangles; ++i) {
    const { a, b, c } = triangles[i];
    x[i] = a[0];
    x[i + numTriangles] = a[1];
    x[i + numTriangles * 2] = b[0];
    x[i + numTriangles * 3] = b[1];
    x[i + numTriangles * 4] = c[0];
    x[i + numTriangles * 5] = c[1];
  }
  return x;
};

const deserialize = (x: Float64Array): Triangle[] => {
  const triangles: Triangle[] = [];
  for (let i = 0; i < numTriangles; ++i) {
    const a = [x[i], x[i + numTriangles]];
    const b = [x[i + numTriangles * 2], x[i + numTriangles * 3]];
    const c = [x[i + numTriangles * 4], x[i + numTriangles * 5]];
    triangles.push({ a, b, c });
  }
  return triangles;
};

const optimize = (triangles: Triangle[]): Triangle[] => {
  const x = serialize(triangles);
  let params = optimizer.start(numTriangles * 6);
  while (params.optStatus !== "EPConverged")
    params = optimizer.stepUntil(grad, x, params, () => false);
  return deserialize(x);
};

const hueFactor = 360 / numTriangles;

const svg = (triangles: Triangle[]): string => {
  const lines = [
    `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${size} ${size}">`,
  ];
  for (let i = 0; i < numTriangles; ++i) {
    const { a, b, c } = triangles[i];
    const hue = i * hueFactor;
    lines.push(
      `  <polygon points="${a} ${b} ${c}" fill="hsl(${hue} 50% 50%)" />`,
    );
  }
  lines.push("</svg>", "");
  return lines.join("\n");
};

const main = () => {
  const original = init("");
  const optimized = optimize(original);
  Bun.write("out/triangles.svg", svg(optimized));
};

main();
