use std::{
    fs::File,
    io::{self, Write},
};

use rand::prelude::*;
use rand_pcg::Pcg64Mcg;

struct R2 {
    x: f64,
    y: f64,
}

impl R2 {
    fn rand(rng: &mut impl Rng) -> Self {
        Self {
            x: rng.gen_range(0.0..=1.0),
            y: rng.gen_range(0.0..=1.0),
        }
    }
}

struct Triangle {
    a: R2,
    b: R2,
    c: R2,
}

const N: usize = 10;

const HUE_FACTOR: f64 = 360.0 / N as f64;

fn init(seed: u64) -> Vec<Triangle> {
    let mut rng = Pcg64Mcg::seed_from_u64(seed);
    let mut triangles = Vec::with_capacity(N);
    for _ in 0..N {
        let a = R2::rand(&mut rng);
        let b = R2::rand(&mut rng);
        let c = R2::rand(&mut rng);
        triangles.push(Triangle { a, b, c });
    }
    triangles
}

fn write_svg(triangles: &[Triangle]) -> io::Result<()> {
    let mut file = File::create("out.svg")?;
    writeln!(
        file,
        r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1 1">"#
    )?;
    for (i, Triangle { a, b, c }) in triangles.iter().enumerate() {
        writeln!(
            file,
            r#"  <polygon points="{},{} {},{} {},{}" fill="hsl({} 50% 50%)" />"#,
            a.x,
            a.y,
            b.x,
            b.y,
            c.x,
            c.y,
            i as f64 * HUE_FACTOR
        )?;
    }
    writeln!(file, "</svg>")?;
    Ok(())
}

fn main() {
    let triangles = init(0);
    write_svg(&triangles).unwrap();
}
