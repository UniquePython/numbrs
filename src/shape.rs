use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shape {
    dims: Vec<usize>,
}

impl Shape {
    pub fn new(dims: Vec<usize>) -> Self {
        if dims.iter().any(|&d| d == 0) {
            panic!("Shape dimensions must be non-zero");
        }
        Self { dims }
    }

    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    pub fn size(&self) -> usize {
        if self.dims.is_empty() {
            return 1; // scalar
        }

        self.dims.iter().product()
    }

    pub fn size_for(&self, axis: usize) -> usize {
        if axis >= self.rank() {
            panic!(
                "Axis {axis} is out of bounds: [0, {len})",
                len = self.rank()
            );
        }
        self.dims[axis]
    }
}

impl fmt::Display for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.dims.is_empty() {
            return write!(f, "()");
        }

        let inner: String = self
            .dims
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(", ");

        write!(f, "({})", inner)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn construct_and_rank() {
        let s: Shape = Shape::new(vec![2, 3, 4]);
        assert_eq!(s.rank(), 3);
    }

    #[test]
    fn dims_returns_slice() {
        let s: Shape = Shape::new(vec![5, 6]);
        assert_eq!(s.dims(), &[5, 6]);
    }

    #[test]
    fn size_computes_product() {
        let s: Shape = Shape::new(vec![2, 3, 4]);
        assert_eq!(s.size(), 24);
    }

    #[test]
    fn size_scalar_case() {
        let s: Shape = Shape::new(vec![]);
        assert_eq!(s.size(), 1);
        assert_eq!(s.rank(), 0);
    }

    #[test]
    fn dim_size_works() {
        let s: Shape = Shape::new(vec![10, 20, 30]);
        assert_eq!(s.size_for(1), 20);
    }

    #[test]
    #[should_panic(expected = "Axis")]
    fn dim_size_out_of_bounds_panics() {
        let s: Shape = Shape::new(vec![1, 2]);
        s.size_for(5);
    }

    #[test]
    #[should_panic(expected = "non-zero")]
    fn zero_dimension_panics() {
        Shape::new(vec![3, 0, 4]);
    }

    #[test]
    fn display_formats_correctly() {
        assert_eq!(Shape::new(vec![3, 4]).to_string(), "(3, 4)");
        assert_eq!(Shape::new(vec![5]).to_string(), "(5)");
        assert_eq!(Shape::new(vec![]).to_string(), "()");
    }
}
