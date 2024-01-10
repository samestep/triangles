#!/usr/bin/env bun

import {
  Real,
  Vec,
  add,
  compile,
  fn,
  mul,
  neg,
  sqrt,
  sub,
  vec,
  vjp,
} from "rose";
import seedrandom from "seedrandom";
import * as lbfgs from "./lbfgs.js";

const sqr = (x: Real): Real => mul(x, x);

const vsub = (a: Real[], b: Real[]): Real[] => a.map((x, i) => sub(x, b[i]));

const dot = (a: Real[], b: Real[]): Real =>
  a.map((x, i) => mul(x, b[i])).reduce((a, b) => add(a, b));

const norm = (v: Real[]): Real => sqrt(dot(v, v));

const area = (ab: Real, bc: Real, ca: Real): Real => {
  const one = add(add(ab, bc), ca);
  const two = add(add(neg(ab), bc), ca);
  const three = add(sub(ab, bc), ca);
  const four = sub(add(ab, bc), ca);

  return mul(0.25, sqrt(mul(mul(mul(one, two), three), four)));
};

type Vec2 = number[];

interface Triangle {
  a: Vec2;
  b: Vec2;
  c: Vec2;
}

const numTriangles = 10;

const fanout = fn([Real], Vec(numTriangles, Real), (x) =>
  vec(numTriangles, Real, () => x),
);

const sum = fn([Vec(numTriangles, Real)], Real, (xs) =>
  vjp(fanout)(0).grad(xs),
);

const lagrangian = fn(
  [
    {
      ax: Vec(numTriangles, Real),
      ay: Vec(numTriangles, Real),
      bx: Vec(numTriangles, Real),
      by: Vec(numTriangles, Real),
      cx: Vec(numTriangles, Real),
      cy: Vec(numTriangles, Real),
    },
  ],
  Real,
  ({ ax, ay, bx, by, cx, cy }) => {
    return sum(
      vec(numTriangles, Real, (i) => {
        const a = [ax[i], ay[i]];
        const b = [bx[i], by[i]];
        const c = [cx[i], cy[i]];

        const ab = norm(vsub(b, a));
        const bc = norm(vsub(c, b));
        const ca = norm(vsub(a, c));

        return add(
          sqr(sub(area(ab, bc, ca), 0.01)),
          add(add(sqr(sub(ab, bc)), sqr(sub(bc, ca))), sqr(sub(ca, ab))),
        );
      }),
    );
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

const grad: lbfgs.Fn = (x: Float64Array, dx: Float64Array): number => {
  const { z, ax, ay, bx, by, cx, cy } = compiled({
    ax: x.subarray(0, numTriangles) as any,
    ay: x.subarray(numTriangles, numTriangles * 2) as any,
    bx: x.subarray(numTriangles * 2, numTriangles * 3) as any,
    by: x.subarray(numTriangles * 3, numTriangles * 4) as any,
    cx: x.subarray(numTriangles * 4, numTriangles * 5) as any,
    cy: x.subarray(numTriangles * 5, numTriangles * 6) as any,
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
    const a = [rng(), rng()];
    const b = [rng(), rng()];
    const c = [rng(), rng()];
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
  const cfg: lbfgs.Config = {
    m: 17,
    armijo: 0.001,
    wolfe: 0.9,
    minInterval: 1e-9,
    maxSteps: 10,
    epsd: 1e-11,
  };
  const x = serialize(triangles);
  const state = lbfgs.firstStep(cfg, grad, x);
  let i = 0;
  lbfgs.stepUntil(cfg, grad, x, state, () => {
    if (++i >= 100) return true;
  });
  return deserialize(x);
};

const hueFactor = 360 / numTriangles;

const svg = (triangles: Triangle[]): string => {
  const lines = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1 1">'];
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
  Bun.write("out.svg", svg(optimized));
};

main();
