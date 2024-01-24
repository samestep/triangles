import * as fs from "fs/promises";
import { exact, run } from "./common.js";
import { mMax, n, r, seed } from "./config.js";

const name = "exact";
for (let m = 3; m <= mMax; ++m) {
  const millis = await run({
    m,
    r,
    n,
    seed,
    minkowski: exact(m),
    name,
  });
  await fs.appendFile(`${name}.csv`, `${name},${m},${r},${n},${millis}\n`);
}
