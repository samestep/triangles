// adapted from:
// https://github.com/penrose/penrose/blob/d7c1ef4be11ac0251f026c755039bccc05818303/packages/core/src/engine/Autodiff.ts
// https://github.com/penrose/penrose/blob/d7c1ef4be11ac0251f026c755039bccc05818303/packages/core/src/engine/AutodiffFunctions.ts
// https://github.com/penrose/penrose/blob/d7c1ef4be11ac0251f026c755039bccc05818303/packages/core/src/engine/Builtins.ts
// https://github.com/penrose/penrose/blob/f9092321d591497cd398bafb6f5fa761e933dcfb/packages/core/src/lib/Functions.ts
// https://github.com/penrose/penrose/blob/4621bb5f502b7e14bf70c828176f282ca2acd9af/packages/core/src/lib/Minkowski.ts
// https://github.com/penrose/penrose/blob/d7c1ef4be11ac0251f026c755039bccc05818303/packages/core/src/lib/Queries.ts

// MIT License

// Copyright (c) 2017

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

import {
  Dual,
  Real,
  add,
  div,
  fn,
  gt,
  lt,
  mul,
  neg,
  sqrt as roseSqrt,
  select,
  sub,
} from "rose";
import { run } from "./common.js";

const epsilon = 1e-5;
const EPS_DENOM = epsilon; // Avoid divide-by-zero in denominator

const squared = (x: Real): Real => mul(x, x);

const max = (a: Real, b: Real) => select(gt(a, b), Real, a, b);

const maxN = (xs: Real[]): Real => {
  // interestingly, special-casing 1 and 2 args like this actually affects the
  // gradient by a nontrivial amount in some cases
  switch (xs.length) {
    case 1: {
      return xs[0];
    }
    case 2: {
      return max(xs[0], xs[1]);
    }
    default: {
      return xs.reduce(max);
    }
  }
};

const sqrt = fn([Real], Real, (x) => roseSqrt(x));
sqrt.jvp = fn([Dual], Dual, ({ re: x, du: dx }) => {
  const y = sqrt(x);
  // NOTE: Watch out for divide by zero in 1 / [2 sqrt(x)]
  const dy = mul(dx, div(1 / 2, max(epsilon, y)));
  return { re: y, du: dy };
});

const ops = {
  /**
   * Return the sum of vectors `v1, v2`.
   */
  vadd: (v1: Real[], v2: Real[]): Real[] => {
    if (v1.length !== v2.length) {
      throw Error("expected vectors of same length");
    }

    const res = v1.map((x, i) => add(x, v2[i]));
    return res;
  },

  /**
   * Return the difference of vectors `v1` and `v2`.
   */
  vsub: (v1: Real[], v2: Real[]): Real[] => {
    if (v1.length !== v2.length) {
      throw Error("expected vectors of same length");
    }

    const res = v1.map((x, i) => sub(x, v2[i]));
    return res;
  },

  /**
   * Return the Euclidean norm squared of vector `v`.
   */
  vnormsq: (v: Real[]): Real => {
    const res = v.map((e) => squared(e));
    return res.reduce((x: Real, y) => add(x, y), 0);
    // Note (performance): the use of 0 adds an extra +0 to the comp graph, but lets us prevent undefined if the list is empty
  },

  /**
   * Return the Euclidean norm of vector `v`.
   */
  vnorm: (v: Real[]): Real => {
    const res = ops.vnormsq(v);
    return sqrt(res);
  },

  /**
   * Return the vector `v` multiplied by scalar `c`.
   */
  vmul: (c: Real, v: Real[]): Real[] => {
    return v.map((e) => mul(c, e));
  },

  /**
   * Return the vector `v` divided by scalar `c`.
   */
  vdiv: (v: Real[], c: Real): Real[] => {
    return v.map((e) => div(e, c));
  },

  /**
   * Return the vector `v`, normalized.
   */
  vnormalize: (v: Real[]): Real[] => {
    const vsize = add(ops.vnorm(v), EPS_DENOM);
    return ops.vdiv(v, vsize);
  },

  /**
   * Return the dot product of vectors `v1, v2`.
   * Note: if you want to compute a norm squared, use `vnormsq` instead, it generates a smaller computational graph
   */
  vdot: (v1: Real[], v2: Real[]): Real => {
    if (v1.length !== v2.length) {
      throw Error("expected vectors of same length");
    }

    const res = v1.map((x, i) => mul(x, v2[i]));
    return res.reduce((x: Real, y) => add(x, y), 0);
  },

  /**
   * Rotate a 2D point `[x, y]` by 90 degrees counterclockwise.
   */
  rot90: ([x, y]: Real[]): Real[] => {
    return [neg(y), x];
  },
};

/*
  float msign(in float x) { return (x<0.0)?-1.0:1.0; }
*/
const msign = (x: Real): Real => {
  return select(lt(x, 0), Real, -1, 1);
};

/**
 * Return outward unit normal vector to `lineSegment` with respect to `insidePoint`.
 * @param lineSegment Two points defining the line segment.
 * @param insidePoint Any point inside of the half-plane.
 */
const outwardUnitNormal = (
  lineSegment: Real[][],
  insidePoint: Real[],
): Real[] => {
  const normal = ops.vnormalize(
    ops.rot90(ops.vsub(lineSegment[1], lineSegment[0])),
  );
  const insideValue = ops.vdot(ops.vsub(insidePoint, lineSegment[0]), normal);
  return ops.vmul(neg(msign(insideValue)), normal);
};

/**
 * Return value of the Signed Distance Function (SDF) of a half-plane evaluated at the origin.
 * @param lineSegment Two points defining a side of the first polygon.
 * @param otherPoints All vertices of the second polygon.
 * @param insidePoint Point inside of the half-plane.
 * @param padding Padding added to the half-plane.
 */
const halfPlaneSDF = (
  lineSegment: Real[][],
  otherPoints: Real[][],
  insidePoint: Real[],
  padding: Real,
): Real => {
  const normal = outwardUnitNormal(lineSegment, insidePoint);
  const alpha = ops.vdot(normal, lineSegment[0]);
  const alphaOther = maxN(otherPoints.map((p) => ops.vdot(normal, p)));
  return add(neg(add(alpha, alphaOther)), padding);
};

/**
 * Return value of one-sided Signed Distance Function (SDF) of the Minkowski sum of two polygons `p1` and `p2` evaluated at the origin.
 * Only half-planes related to sides of the first polygon `p1` are considered.
 * @param p1 Sequence of points defining the first polygon.
 * @param p2 Sequence of points defining the second polygon.
 * @param padding Padding around the Minkowski sum.
 */
const convexPolygonMinkowskiSDFOneSided = (
  p1: Real[][],
  p2: Real[][],
  padding: Real,
): Real => {
  const center = ops.vdiv(p1.reduce(ops.vadd), p1.length);
  // Create a list of all sides given by two subsequent vertices
  const sides = Array.from({ length: p1.length }, (_, key) => key).map((i) => [
    p1[i],
    p1[i > 0 ? i - 1 : p1.length - 1],
  ]);
  const sdfs = sides.map((s: Real[][]) => halfPlaneSDF(s, p2, center, padding));
  return maxN(sdfs);
};

/**
 * Return value of the Signed Distance Function (SDF) of the Minkowski sum of two polygons `p1` and `p2` evaluated at the origin.
 * @param p1 Sequence of points defining the first polygon.
 * @param p2 Sequence of points defining the second polygon.
 * @param padding Padding around the Minkowski sum.
 */
const convexPolygonMinkowskiSDF = (
  p1: Real[][],
  p2: Real[][],
  padding: Real,
): Real => {
  return max(
    convexPolygonMinkowskiSDFOneSided(p1, p2, padding),
    convexPolygonMinkowskiSDFOneSided(p2, p1, padding),
  );
};

await run({
  m: 3,
  r: 12 / Math.sqrt(3),
  n: 100,
  seed: "",
  minkowski: (p, q) =>
    convexPolygonMinkowskiSDF(
      p,
      q.map(([x, y]) => [neg(x), neg(y)]),
      0,
    ),
  name: "approx",
});
