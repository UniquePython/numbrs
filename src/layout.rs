use crate::{shape::Shape, strides::Strides};

pub fn index_to_offset(indices: &[usize], strides: &Strides, shape: &Shape) -> usize {
    if indices.len() != shape.rank() {
        panic!(
            "Index rank mismatch: expected {}, got {}",
            shape.rank(),
            indices.len()
        );
    }

    let mut offset: usize = 0;

    for (axis, (idx, stride)) in indices.iter().zip(strides.as_slice()).enumerate() {
        if *idx >= shape.dim_size(axis) {
            panic!(
                "Index {idx} out of bounds for axis {axis}: [0, {})",
                shape.dim_size(axis)
            );
        }
        offset += idx * stride;
    }

    offset
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{shape::Shape, strides::Strides};

    #[test]
    fn scalar() {
        let s: Shape = Shape::new(vec![]);
        let st: Strides = Strides::from_shape_c(&s);

        assert_eq!(index_to_offset(&[], &st, &s), 0);
    }

    #[test]
    fn basic_2d() {
        let s: Shape = Shape::new(vec![3, 4]);
        let st: Strides = Strides::from_shape_c(&s);

        assert_eq!(index_to_offset(&[1, 2], &st, &s), 6);
    }

    #[test]
    fn basic_3d() {
        let s: Shape = Shape::new(vec![2, 3, 4]);
        let st: Strides = Strides::from_shape_c(&s);

        assert_eq!(index_to_offset(&[1, 2, 3], &st, &s), 23);
    }

    #[test]
    fn last_valid_index_1d() {
        let s: Shape = Shape::new(vec![5]);
        let st: Strides = Strides::from_shape_c(&s);
        assert_eq!(index_to_offset(&[4], &st, &s), 4);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn oob() {
        let s: Shape = Shape::new(vec![2, 2]);
        let st: Strides = Strides::from_shape_c(&s);

        index_to_offset(&[3, 0], &st, &s);
    }
}
