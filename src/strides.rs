use crate::shape::Shape;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Strides {
    strides: Vec<usize>,
}

impl Strides {
    pub fn new(strides: Vec<usize>) -> Self {
        Self { strides }
    }

    fn from_shape(shape: &Shape, iter: impl Iterator<Item = usize>) -> Self {
        let dims: &[usize] = shape.dims();
        let mut strides: Vec<usize> = vec![0; dims.len()];

        if dims.is_empty() {
            return Self { strides };
        }

        let mut acc: usize = 1;
        for i in iter {
            strides[i] = acc;
            acc *= dims[i];
        }

        Self { strides }
    }

    pub fn from_shape_c(shape: &Shape) -> Self {
        let dims_len: usize = shape.dims().len();
        Self::from_shape(shape, (0..dims_len).rev())
    }

    pub fn from_shape_f(shape: &Shape) -> Self {
        let dims_len: usize = shape.dims().len();
        Self::from_shape(shape, 0..dims_len)
    }

    pub fn as_slice(&self) -> &[usize] {
        &self.strides
    }

    pub fn num_strides(&self) -> usize {
        self.strides.len()
    }

    pub fn stride_for(&self, axis: usize) -> usize {
        if axis >= self.num_strides() {
            panic!(
                "Axis {axis} is out of bounds: [0, {len})",
                len = self.num_strides()
            );
        }
        self.strides[axis]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shape::Shape;

    #[test]
    fn c_order_2d() {
        let s: Shape = Shape::new(vec![3, 4]);
        let strides: Strides = Strides::from_shape_c(&s);
        assert_eq!(strides.as_slice(), &[4, 1]);
    }

    #[test]
    fn c_order_3d() {
        let s: Shape = Shape::new(vec![2, 3, 4]);
        let strides: Strides = Strides::from_shape_c(&s);
        assert_eq!(strides.as_slice(), &[12, 4, 1]);
    }

    #[test]
    fn f_order_2d() {
        let s: Shape = Shape::new(vec![3, 4]);
        let strides: Strides = Strides::from_shape_f(&s);
        assert_eq!(strides.as_slice(), &[1, 3]);
    }

    #[test]
    fn f_order_3d() {
        let s: Shape = Shape::new(vec![2, 3, 4]);
        let strides: Strides = Strides::from_shape_f(&s);
        assert_eq!(strides.as_slice(), &[1, 2, 6]);
    }

    #[test]
    fn scalar_case() {
        let s: Shape = Shape::new(vec![]);
        let c: Strides = Strides::from_shape_c(&s);
        let f: Strides = Strides::from_shape_f(&s);

        assert_eq!(c.as_slice(), &[]);
        assert_eq!(f.as_slice(), &[]);
        assert_eq!(c.num_strides(), 0);
    }

    #[test]
    fn stride_for_works() {
        let s: Shape = Shape::new(vec![2, 3, 4]);
        let strides: Strides = Strides::from_shape_c(&s);

        assert_eq!(strides.stride_for(0), 12);
        assert_eq!(strides.stride_for(1), 4);
        assert_eq!(strides.stride_for(2), 1);
    }

    #[test]
    #[should_panic(expected = "Axis")]
    fn stride_out_of_bounds() {
        let s: Shape = Shape::new(vec![2, 2]);
        let strides: Strides = Strides::from_shape_c(&s);
        strides.stride_for(5);
    }
}
