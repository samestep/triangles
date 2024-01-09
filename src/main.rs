mod ad;

use ad::{Back, Num, Tape};
use rand::prelude::*;
use rand_pcg::Pcg64Mcg;
use std::{
    fs::File,
    io::{self, Write},
};

struct Vec2<T> {
    x: T,
    y: T,
}

impl Vec2<Num> {
    fn sub(
        &self,
        other: &Self,
        tape: Tape<impl Fn(Back) -> Back>,
    ) -> (Tape<impl Fn(Back) -> Back>, Self) {
        let (tape, x) = tape.sub(self.x, other.x);
        let (tape, y) = tape.sub(self.y, other.y);
        (tape, Self { x, y })
    }

    fn norm(&self, tape: Tape<impl Fn(Back) -> Back>) -> (Tape<impl Fn(Back) -> Back>, Num) {
        let (tape, x2) = tape.mul(self.x, self.x);
        let (tape, y2) = tape.mul(self.y, self.y);
        let (tape, xy) = tape.add(x2, y2);
        tape.sqrt(xy)
    }
}

struct Triangle<T> {
    a: Vec2<T>,
    b: Vec2<T>,
    c: Vec2<T>,
}

impl Triangle<Num> {
    fn area(&self, tape: Tape<impl Fn(Back) -> Back>) -> (Tape<impl Fn(Back) -> Back>, Num) {
        let Self { a, b, c } = self;

        let (tape, ab) = {
            let (tape, ba) = b.sub(a, tape);
            ba.norm(tape)
        };
        let (tape, bc) = {
            let (tape, cb) = c.sub(b, tape);
            cb.norm(tape)
        };
        let (tape, ca) = {
            let (tape, ac) = a.sub(c, tape);
            ac.norm(tape)
        };

        let (tape, one) = {
            let (tape, x) = tape.add(ab, bc);
            tape.add(x, ca)
        };
        let (tape, two) = {
            let (tape, x) = tape.sub(bc, ab);
            tape.add(x, ca)
        };
        let (tape, three) = {
            let (tape, x) = tape.sub(ab, bc);
            tape.add(x, ca)
        };
        let (tape, four) = {
            let (tape, x) = tape.add(ab, bc);
            tape.sub(x, ca)
        };

        let (tape, prod) = {
            let (tape, x) = tape.mul(one, two);
            let (tape, y) = tape.mul(x, three);
            tape.mul(y, four)
        };
        let (mut tape, sqrt) = tape.sqrt(prod);
        let quarter = tape.var(0.25);
        tape.mul(quarter, sqrt)
    }
}

/// number of triangles
const TRIANGLES: usize = 10;

/// number of degrees of freedom
const N: usize = TRIANGLES * 3 * 2;

const HUE_FACTOR: f64 = 360.0 / TRIANGLES as f64;

fn init(seed: u64) -> Vec<f64> {
    let mut rng = Pcg64Mcg::seed_from_u64(seed);
    let mut v = Vec::with_capacity(N);
    for _ in 0..N {
        v.push(rng.gen_range(0.0..=1.0));
    }
    v
}

fn deserialize<T: Copy>(v: &[T]) -> impl Iterator<Item = Triangle<T>> + '_ {
    v.chunks_exact(6).map(|chunk| Triangle {
        a: Vec2 {
            x: chunk[0],
            y: chunk[1],
        },
        b: Vec2 {
            x: chunk[2],
            y: chunk[3],
        },
        c: Vec2 {
            x: chunk[4],
            y: chunk[5],
        },
    })
}

fn optimize(v: &mut [f64]) {
    let mut tape = ad::tape();
    let nums: Vec<Num> = v.iter().map(|&x| tape.var(x)).collect();
    let triangle = deserialize(&nums).next().unwrap();
    triangle.area(tape);
}

fn write_svg(v: &[f64]) -> io::Result<()> {
    let mut file = File::create("out.svg")?;
    writeln!(
        file,
        r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1 1">"#
    )?;
    for (i, Triangle { a, b, c }) in deserialize(v).enumerate() {
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
    let mut v = init(0);
    optimize(&mut v);
    write_svg(&v).unwrap();
}
