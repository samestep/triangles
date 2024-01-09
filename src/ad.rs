pub struct Back {
    vals: Vec<f64>,
    grad: Vec<f64>,
}

impl Back {
    fn get(&self, x: Num) -> f64 {
        self.vals[x.i]
    }
}

pub struct Tape<F: Fn(Back) -> Back> {
    vals: Vec<f64>,
    grad: F,
}

#[derive(Clone, Copy)]
pub struct Num {
    i: usize,
}

pub fn tape() -> Tape<impl Fn(Back) -> Back> {
    Tape {
        vals: Vec::new(),
        grad: |back| back,
    }
}

impl<F: Fn(Back) -> Back> Tape<F> {
    fn get(&self, x: Num) -> f64 {
        self.vals[x.i]
    }

    pub fn var(&mut self, val: f64) -> Num {
        let i = self.vals.len();
        self.vals.push(val);
        Num { i }
    }

    pub fn add(mut self, x: Num, y: Num) -> (Tape<impl Fn(Back) -> Back>, Num) {
        let z = self.var(self.get(x) + self.get(y));
        let Self { vals, grad } = self;
        (
            Tape {
                vals,
                grad: move |mut back| {
                    let dz = back.grad[z.i];
                    back.grad[x.i] += dz;
                    back.grad[y.i] += dz;
                    grad(back)
                },
            },
            z,
        )
    }

    pub fn sub(mut self, x: Num, y: Num) -> (Tape<impl Fn(Back) -> Back>, Num) {
        let z = self.var(self.get(x) - self.get(y));
        let Self { vals, grad } = self;
        (
            Tape {
                vals,
                grad: move |mut back| {
                    let dz = back.grad[z.i];
                    back.grad[x.i] += dz;
                    back.grad[y.i] -= dz;
                    grad(back)
                },
            },
            z,
        )
    }

    pub fn mul(mut self, x: Num, y: Num) -> (Tape<impl Fn(Back) -> Back>, Num) {
        let z = self.var(self.get(x) * self.get(y));
        let Self { vals, grad } = self;
        (
            Tape {
                vals,
                grad: move |mut back| {
                    let dz = back.grad[z.i];
                    back.grad[x.i] += dz * back.get(y);
                    back.grad[y.i] += dz * back.get(x);
                    grad(back)
                },
            },
            z,
        )
    }

    pub fn sqrt(mut self, x: Num) -> (Tape<impl Fn(Back) -> Back>, Num) {
        let y = self.var(self.get(x).sqrt());
        let Self { vals, grad } = self;
        (
            Tape {
                vals,
                grad: move |mut back| {
                    let dy = back.grad[y.i];
                    let y = back.get(y);
                    back.grad[x.i] += dy / (y + y);
                    grad(back)
                },
            },
            y,
        )
    }
}
