import { exact, run } from "./common.js";

const m = 3;
await run({ m, minkowski: exact(m), name: "exact" });
