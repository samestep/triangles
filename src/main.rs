mod lbfgs;

use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64Mcg;
use resvg::{
    render,
    tiny_skia::Pixmap,
    usvg::{
        fontdb::Database, Options, PostProcessingSteps, Transform, Tree, TreeParsing, TreePostProc,
    },
};
use std::{
    f64::consts::TAU,
    fmt,
    fs::{create_dir_all, File},
    io::Write,
    iter::once,
    ops::{Add, AddAssign, Div, Mul, Neg, Sub, SubAssign},
    path::Path,
};

#[derive(Clone, Copy, Debug, PartialEq)]
struct Vec2 {
    x: f64,
    y: f64,
}

fn vec2(x: f64, y: f64) -> Vec2 {
    Vec2 { x, y }
}

impl Neg for Vec2 {
    type Output = Vec2;

    fn neg(self) -> Vec2 {
        vec2(-self.x, -self.y)
    }
}

impl Add for Vec2 {
    type Output = Vec2;

    fn add(self, rhs: Vec2) -> Vec2 {
        vec2(self.x + rhs.x, self.y + rhs.y)
    }
}

impl Sub for Vec2 {
    type Output = Vec2;

    fn sub(self, rhs: Vec2) -> Vec2 {
        vec2(self.x - rhs.x, self.y - rhs.y)
    }
}

impl Mul<Vec2> for f64 {
    type Output = Vec2;

    fn mul(self, rhs: Vec2) -> Vec2 {
        vec2(self * rhs.x, self * rhs.y)
    }
}

impl Mul<f64> for Vec2 {
    type Output = Vec2;

    fn mul(self, rhs: f64) -> Vec2 {
        vec2(self.x * rhs, self.y * rhs)
    }
}

impl Div<f64> for Vec2 {
    type Output = Vec2;

    fn div(self, rhs: f64) -> Vec2 {
        vec2(self.x / rhs, self.y / rhs)
    }
}

impl AddAssign for Vec2 {
    fn add_assign(&mut self, rhs: Vec2) {
        *self = *self + rhs;
    }
}

impl SubAssign for Vec2 {
    fn sub_assign(&mut self, rhs: Vec2) {
        *self = *self - rhs;
    }
}

fn dot(u: Vec2, v: Vec2) -> f64 {
    u.x * v.x + u.y * v.y
}

fn cross(u: Vec2, v: Vec2) -> f64 {
    u.x * v.y - u.y * v.x
}

const SIZE: f64 = 100.;

fn init(seed: u64, n: usize) -> Vec<f64> {
    let mut rng = Pcg64Mcg::seed_from_u64(seed);
    let mut coords: Vec<_> = (0..(2 * n)).map(|_| rng.gen_range(0.0..=SIZE)).collect();
    coords.extend((0..n).map(|_| rng.gen_range(0.0..360.0)));
    coords
}

fn split(coords: &[f64]) -> (&[f64], &[f64], &[f64]) {
    let n = coords.len() / 3;
    let (x, rest) = coords.split_at(n);
    let (y, h) = rest.split_at(n);
    (x, y, h)
}

fn split_mut(coords: &mut [f64]) -> (&mut [f64], &mut [f64], &mut [f64]) {
    let n = coords.len() / 3;
    let (x, rest) = coords.split_at_mut(n);
    let (y, h) = rest.split_at_mut(n);
    (x, y, h)
}

#[derive(Clone, Copy)]
struct Config {
    m: usize,
    r: f64,
    n: usize,
    seed: u64,
}

fn polygon(cfg: Config, x: f64, y: f64, theta: f64) -> (Vec<Vec2>, Vec<(f64, f64)>) {
    (0..cfg.m)
        .map(|i| {
            let theta = theta + TAU * (i as f64 / cfg.m as f64);
            let sin = theta.sin();
            let cos = theta.cos();
            (vec2(x + cfg.r * cos, y + cfg.r * sin), (sin, cos))
        })
        .unzip()
}

// https://cp-algorithms.com/geometry/minkowski.html

fn reorder(p: &[Vec2]) -> Vec<usize> {
    let n = p.len();
    let mut pos = 0;
    for i in 1..n {
        if p[i].y < p[pos].y || (p[i].y == p[pos].y && p[i].x < p[pos].x) {
            pos = i;
        }
    }
    (pos..n).chain(0..=pos).chain(once((pos + 1) % n)).collect()
}

fn minkowski_sum(p: &[Vec2], q: &[Vec2]) -> Vec<(usize, usize)> {
    let m = p.len();
    let n = q.len();
    let is = reorder(p);
    let js = reorder(q);
    let mut r = vec![];
    let mut i = 0;
    let mut j = 0;
    while i < m || j < n {
        let ii = is[i];
        let jj = js[j];
        r.push((ii, jj));
        let z = cross(p[is[i + 1]] - p[ii], q[js[j + 1]] - q[jj]);
        if z >= 0. && i < m {
            i += 1;
        }
        if z <= 0. && j < n {
            j += 1;
        }
    }
    r
}

struct SignedDist {
    d: f64,
    i: usize,
    di: Vec2,
    j: usize,
    dj: Vec2,
}

// https://iquilezles.org/articles/distfunctions2d/
fn sd_polygon(v: &[Vec2], p: Vec2) -> SignedDist {
    let n = v.len();
    let u = p - v[0];
    let mut d = dot(u, u);
    let mut s = 1.0;
    let mut i = 0;
    let mut j = n - 1;
    let mut ii = i;
    let mut di = -2. * u;
    let mut jj = j;
    let mut dj = vec2(0., 0.);
    while i < n {
        let e = v[j] - v[i];
        let w = p - v[i];
        let we = dot(w, e);
        let ee = dot(e, e);
        let r = we / ee;
        let rc = r.clamp(0.0, 1.0);
        let b = w - e * rc;
        let bb = dot(b, b);
        if bb < d {
            d = bb;
            ii = i;
            jj = j;
            let db = 2. * b;
            let mut dw = db;
            let mut de = -db * rc;
            let drc = -dot(e, db);
            let dr = if (0.0..=1.0).contains(&r) { drc } else { 0. };
            let dwe = dr / ee;
            let dee = dwe * -r;
            de += dee * 2. * e;
            dw += dwe * e;
            de += dwe * w;
            di = -dw - de;
            dj = de;
        }
        let c = [p.y >= v[i].y, p.y < v[j].y, e.x * w.y > e.y * w.x];
        if c.iter().all(|&a| a) || c.iter().all(|&a| !a) {
            s *= -1.0;
        }
        j = i;
        i += 1;
    }
    let z = s * d.sqrt();
    let w = 2. * z;
    SignedDist {
        d: z,
        i: ii,
        di: di / w,
        j: jj,
        dj: dj / w,
    }
}

fn val_and_grad(cfg: Config, coords: &[f64], grad: &mut [f64]) -> f64 {
    grad.fill(0.);
    let (xs, ys, thetas) = split(coords);
    let (polys, trigs): (Vec<_>, Vec<_>) = xs
        .iter()
        .zip(ys)
        .zip(thetas)
        .map(|((&x, &y), &theta)| polygon(cfg, x, y, theta))
        .unzip();
    let n = polys.len();
    let mut dpolys = vec![vec![vec2(0., 0.); cfg.m]; n];
    let mut fx = 0.;
    for (poly, dpoly) in polys.iter().zip(&mut dpolys) {
        for (&Vec2 { x, y }, Vec2 { x: dx, y: dy }) in poly.iter().zip(dpoly) {
            if x < 0. {
                fx += x * x;
                *dx += 2. * x;
            }
            if x > SIZE {
                let z = x - SIZE;
                fx += z * z;
                *dx += 2. * z;
            }
            if y < 0. {
                fx += y * y;
                *dy += 2. * y;
            }
            if y > SIZE {
                let z = y - SIZE;
                fx += z * z;
                *dy += 2. * z;
            }
        }
    }
    for i in 0..n {
        for j in (i + 1)..n {
            let p = &polys[i];
            let q: Vec<_> = polys[j].iter().map(|&v| -v).collect();
            let indices = minkowski_sum(p, &q);
            let r: Vec<_> = indices.iter().map(|&(ii, jj)| p[ii] + q[jj]).collect();
            let dist = sd_polygon(&r, vec2(0., 0.));
            let z = -dist.d;
            if z > 0. {
                fx += z * z;
                let di = 2. * z * dist.di;
                let dj = 2. * z * dist.dj;
                let (ii, ij) = indices[dist.i];
                dpolys[i][ii] += di;
                dpolys[j][ij] -= di;
                let (ji, jj) = indices[dist.j];
                dpolys[i][ji] += dj;
                dpolys[j][jj] -= dj;
            }
        }
    }
    let (dxs, dys, dthetas) = split_mut(grad);
    for i in 0..n {
        let trig = &trigs[i];
        let dpoly = &dpolys[i];
        for j in 0..cfg.m {
            let (sin, cos) = trig[j];
            let dv = dpoly[j];
            dxs[i] += dv.x;
            dys[i] += dv.y;
            dthetas[i] += cfg.r * dot(dv, vec2(-sin, cos));
        }
    }
    fx
}

fn optimize(
    config: Config,
    coords: &mut [f64],
    mut callback: impl FnMut(Option<&lbfgs::Info>, &[f64]),
) -> f64 {
    callback(None, coords);
    let cfg = lbfgs::Config {
        m: 17,
        armijo: 0.001,
        wolfe: 0.9,
        min_interval: 1e-9,
        max_steps: 10,
        epsd: 1e-11,
    };
    let mut state = lbfgs::first_step(
        cfg,
        |coords, grad| val_and_grad(config, coords, grad),
        coords,
    );
    callback(None, coords);
    let mut fx = f64::NAN;
    lbfgs::step_until(
        cfg,
        |coords, grad| val_and_grad(config, coords, grad),
        coords,
        &mut state,
        |info| {
            callback(Some(&info), info.x);
            if info.fx == fx {
                Some(())
            } else {
                fx = info.fx;
                None
            }
        },
    );
    fx
}

// https://github.com/penrose/penrose/blob/7c1978f4e33498828d6893d7d8f9257d2f1f839b/packages/core/src/utils/Util.ts#L415-L450
fn hsv_to_rgb(h0: f64, s0: f64, v0: f64) -> (f64, f64, f64) {
    fn hsv2rgb(r1: f64, g1: f64, b1: f64, m: f64) -> (f64, f64, f64) {
        (
            (255. * (r1 + m)).round(),
            (255. * (g1 + m)).round(),
            (255. * (b1 + m)).round(),
        )
    }

    let h = (h0 % 360.) + if h0 < 0. { 360. } else { 0. };
    let s = s0 / 100.0;
    let v = v0 / 100.0;
    let c = v * s;
    let x = c * (1. - (((h / 60.) % 2.) - 1.).abs());
    let m = v - c;

    if h < 60. {
        hsv2rgb(c, x, 0., m)
    } else if h < 120. {
        hsv2rgb(x, c, 0., m)
    } else if h < 180. {
        hsv2rgb(0., c, x, m)
    } else if h < 240. {
        hsv2rgb(0., x, c, m)
    } else if h < 300. {
        hsv2rgb(x, 0., c, m)
    } else {
        hsv2rgb(c, 0., x, m)
    }
}

fn draw(w: &mut impl fmt::Write, cfg: Config, coords: &[f64]) -> fmt::Result {
    writeln!(
        w,
        r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {SIZE} {SIZE}">"#,
    )?;
    let (xs, ys, thetas) = split(coords);
    for ((&x, &y), &theta) in xs.iter().zip(ys).zip(thetas) {
        let (r, g, b) = hsv_to_rgb(theta, 60., 100.);
        writeln!(w, r##"  <polygon fill="rgb({r} {g} {b})" points=""##)?;
        let (points, _) = polygon(cfg, x, y, theta);
        for Vec2 { x, y } in points {
            writeln!(w, "      {},{}", x, y)?;
        }
        writeln!(w, r#"    " />"#)?;
    }
    writeln!(w, "</svg>")?;
    Ok(())
}

fn rasterize(svg: &str, scale: f32) -> Pixmap {
    let mut tree = Tree::from_str(svg, &Options::default()).unwrap();
    tree.postprocess(PostProcessingSteps::default(), &Database::new());
    let mut pixmap = Pixmap::new(
        (scale * tree.size.width()) as u32,
        (scale * tree.size.height()) as u32,
    )
    .unwrap();
    render(
        &tree,
        Transform::from_scale(scale, scale),
        &mut pixmap.as_mut(),
    );
    pixmap
}

fn run(dir: &Path, cfg: Config) {
    let dir_frames = dir.join(format!("{}-{}-{}", cfg.m, cfg.n, cfg.seed));
    create_dir_all(&dir_frames).unwrap();
    let scale = 10.;
    let mut i: usize = 0;
    let mut coords = init(cfg.seed, cfg.n);
    let fx = optimize(cfg, &mut coords, |info, coords| {
        if i.count_ones() < 2 {
            print!("i = {i}");
            if let Some(info) = info {
                println!(", fx = {}", info.fx);
            } else {
                println!();
            }
            let mut s = String::new();
            draw(&mut s, cfg, coords).unwrap();
            File::create(dir_frames.join(format!("{i}.svg")))
                .unwrap()
                .write_all(s.as_bytes())
                .unwrap();
            rasterize(&s, scale)
                .save_png(dir_frames.join(format!("{i}.png")))
                .unwrap();
        }
        i += 1;
    });
    i -= 1;
    println!("i = {i}, fx = {fx}");
    let mut s = String::new();
    draw(&mut s, cfg, &coords).unwrap();
    File::create(dir_frames.join(format!("{i}.svg")))
        .unwrap()
        .write_all(s.as_bytes())
        .unwrap();
    rasterize(&s, scale)
        .save_png(dir_frames.join(format!("{i}.png")))
        .unwrap();
}

fn main() {
    let dir = Path::new("out");
    let cfg = Config {
        m: 3,
        r: (3.0f64.sqrt() / 3.) * 12.,
        n: 100,
        seed: 0,
    };
    run(dir, cfg);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_minkowski_sum() {
        assert_eq!(
            minkowski_sum(
                &[vec2(-2., 2.), vec2(-4., 2.), vec2(-3., 1.)],
                &[vec2(2., 3.), vec2(2., 1.), vec2(4., 1.), vec2(4., 3.)],
            ),
            vec![(2, 1), (2, 2), (0, 2), (0, 3), (1, 0), (1, 1)],
        )
    }
}
