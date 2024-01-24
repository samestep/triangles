import { exact, run } from "./common.js";

const m = 3;
await run({
  m,
  r: 12 / Math.sqrt(3),
  n: 100,
  seed: "",
  minkowski: exact(m),
  name: "exact",
});
