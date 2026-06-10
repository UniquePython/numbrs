use crate::{buffer::RawBuffer, layout::index_to_offset, shape::Shape, strides::Strides};

pub struct NdArray<T> {
    data: RawBuffer<T>,
    offset: usize,
    shape: Shape,
    strides: Strides,
}

impl<T> NdArray<T> {
    pub fn new(data: Vec<T>, shape: Shape) -> Self {
        if data.len() != shape.size() {
            panic!(
                "Expected {} elements in data based on shape {shape}, found {}",
                shape.size(),
                data.len()
            )
        }

        let strides: Strides = Strides::from_shape_c(&shape);

        Self {
            data: RawBuffer::new(data),
            offset: 0,
            shape,
            strides,
        }
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn rank(&self) -> usize {
        self.shape.rank()
    }

    pub fn size(&self) -> usize {
        self.shape.size()
    }

    pub fn size_for(&self, axis: usize) -> usize {
        self.shape.size_for(axis)
    }

    pub fn strides(&self) -> &Strides {
        &self.strides
    }

    pub fn stride_for(&self, axis: usize) -> usize {
        self.strides.stride_for(axis)
    }

    pub fn get(&self, indices: &[usize]) -> T
    where
        T: Clone,
    {
        let local_offset: usize = index_to_offset(indices, &self.strides, &self.shape);

        self.data.get(self.offset + local_offset).clone()
    }

    pub fn set(&mut self, indices: &[usize], value: T) {
        let local_offset: usize = index_to_offset(indices, &self.strides, &self.shape);

        self.data.set(self.offset + local_offset, value);
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::shape::Shape;

    #[test]
    fn new_creates_1d_array() {
        let arr: NdArray<i32> = NdArray::new(vec![1, 2, 3], Shape::new(vec![3]));

        assert_eq!(arr.rank(), 1);
        assert_eq!(arr.size(), 3);
        assert_eq!(arr.shape(), &Shape::new(vec![3]));
    }

    #[test]
    fn new_creates_2d_array() {
        let arr: NdArray<i32> = NdArray::new(vec![1, 2, 3, 4, 5, 6], Shape::new(vec![2, 3]));

        assert_eq!(arr.rank(), 2);
        assert_eq!(arr.size(), 6);
        assert_eq!(arr.size_for(0), 2);
        assert_eq!(arr.size_for(1), 3);
    }

    #[test]
    fn new_creates_3d_array() {
        let arr: NdArray<i32> = NdArray::new((0..24).collect(), Shape::new(vec![2, 3, 4]));

        assert_eq!(arr.rank(), 3);
        assert_eq!(arr.size(), 24);
        assert_eq!(arr.size_for(0), 2);
        assert_eq!(arr.size_for(1), 3);
        assert_eq!(arr.size_for(2), 4);
    }

    #[test]
    fn shape_returns_original_shape() {
        let shape: Shape = Shape::new(vec![2, 3, 4]);

        let arr: NdArray<i32> = NdArray::new((0..24).collect(), shape.clone());

        assert_eq!(arr.shape(), &shape);
    }

    #[test]
    fn strides_match_c_order_for_1d() {
        let arr: NdArray<i32> = NdArray::new(vec![1, 2, 3], Shape::new(vec![3]));

        assert_eq!(arr.stride_for(0), 1);
    }

    #[test]
    fn strides_match_c_order_for_2d() {
        let arr: NdArray<i32> = NdArray::new(vec![1, 2, 3, 4, 5, 6], Shape::new(vec![2, 3]));

        assert_eq!(arr.stride_for(0), 3);
        assert_eq!(arr.stride_for(1), 1);
    }

    #[test]
    fn strides_match_c_order_for_3d() {
        let arr: NdArray<i32> = NdArray::new((0..24).collect(), Shape::new(vec![2, 3, 4]));

        assert_eq!(arr.stride_for(0), 12);
        assert_eq!(arr.stride_for(1), 4);
        assert_eq!(arr.stride_for(2), 1);
    }

    #[test]
    fn get_works_for_1d() {
        let arr: NdArray<i32> = NdArray::new(vec![10, 20, 30, 40], Shape::new(vec![4]));

        assert_eq!(arr.get(&[0]), 10);
        assert_eq!(arr.get(&[1]), 20);
        assert_eq!(arr.get(&[2]), 30);
        assert_eq!(arr.get(&[3]), 40);
    }

    #[test]
    fn get_works_for_2d() {
        let arr: NdArray<i32> = NdArray::new(vec![1, 2, 3, 4, 5, 6], Shape::new(vec![2, 3]));

        assert_eq!(arr.get(&[0, 0]), 1);
        assert_eq!(arr.get(&[0, 1]), 2);
        assert_eq!(arr.get(&[0, 2]), 3);
        assert_eq!(arr.get(&[1, 0]), 4);
        assert_eq!(arr.get(&[1, 1]), 5);
        assert_eq!(arr.get(&[1, 2]), 6);
    }

    #[test]
    fn get_works_for_3d() {
        let arr: NdArray<i32> = NdArray::new((0..24).collect(), Shape::new(vec![2, 3, 4]));

        assert_eq!(arr.get(&[0, 0, 0]), 0);
        assert_eq!(arr.get(&[0, 0, 1]), 1);
        assert_eq!(arr.get(&[0, 1, 0]), 4);
        assert_eq!(arr.get(&[1, 0, 0]), 12);
        assert_eq!(arr.get(&[1, 2, 3]), 23);
    }

    #[test]
    fn set_updates_1d_values() {
        let mut arr: NdArray<i32> = NdArray::new(vec![1, 2, 3], Shape::new(vec![3]));

        arr.set(&[1], 99);

        assert_eq!(arr.get(&[0]), 1);
        assert_eq!(arr.get(&[1]), 99);
        assert_eq!(arr.get(&[2]), 3);
    }

    #[test]
    fn set_updates_2d_values() {
        let mut arr: NdArray<i32> = NdArray::new(vec![1, 2, 3, 4, 5, 6], Shape::new(vec![2, 3]));

        arr.set(&[1, 1], 99);

        assert_eq!(arr.get(&[1, 1]), 99);
    }

    #[test]
    fn set_updates_first_element() {
        let mut arr: NdArray<i32> = NdArray::new(vec![1, 2, 3, 4], Shape::new(vec![2, 2]));

        arr.set(&[0, 0], 42);

        assert_eq!(arr.get(&[0, 0]), 42);
    }

    #[test]
    fn set_updates_last_element() {
        let mut arr: NdArray<i32> = NdArray::new(vec![1, 2, 3, 4], Shape::new(vec![2, 2]));

        arr.set(&[1, 1], 99);

        assert_eq!(arr.get(&[1, 1]), 99);
    }

    #[test]
    fn get_supports_non_copy_types() {
        let arr: NdArray<String> = NdArray::new(
            vec![String::from("a"), String::from("b")],
            Shape::new(vec![2]),
        );

        assert_eq!(arr.get(&[0]), "a");
        assert_eq!(arr.get(&[1]), "b");
    }

    #[test]
    fn set_supports_non_copy_types() {
        let mut arr: NdArray<String> = NdArray::new(
            vec![String::from("a"), String::from("b")],
            Shape::new(vec![2]),
        );

        arr.set(&[1], String::from("z"));

        assert_eq!(arr.get(&[1]), "z");
    }

    #[test]
    #[should_panic(expected = "Expected")]
    fn new_panics_when_data_too_short() {
        NdArray::<i32>::new(vec![1, 2, 3], Shape::new(vec![2, 2]));
    }

    #[test]
    #[should_panic(expected = "Expected")]
    fn new_panics_when_data_too_long() {
        NdArray::<i32>::new(vec![1, 2, 3, 4, 5], Shape::new(vec![2, 2]));
    }

    #[test]
    #[should_panic]
    fn get_panics_when_axis_index_is_out_of_bounds() {
        let arr: NdArray<i32> = NdArray::new(vec![1, 2, 3, 4], Shape::new(vec![2, 2]));

        arr.get(&[2, 0]);
    }

    #[test]
    #[should_panic]
    fn get_panics_when_rank_is_wrong() {
        let arr: NdArray<i32> = NdArray::new(vec![1, 2, 3, 4], Shape::new(vec![2, 2]));

        arr.get(&[0]);
    }

    #[test]
    #[should_panic]
    fn get_panics_when_too_many_indices_are_given() {
        let arr: NdArray<i32> = NdArray::new(vec![1, 2, 3, 4], Shape::new(vec![2, 2]));

        arr.get(&[0, 0, 0]);
    }

    #[test]
    #[should_panic]
    fn set_panics_when_axis_index_is_out_of_bounds() {
        let mut arr: NdArray<i32> = NdArray::new(vec![1, 2, 3, 4], Shape::new(vec![2, 2]));

        arr.set(&[0, 2], 99);
    }

    #[test]
    fn multiple_sets_work_correctly() {
        let mut arr: NdArray<i32> = NdArray::new(vec![0; 9], Shape::new(vec![3, 3]));

        arr.set(&[0, 0], 1);
        arr.set(&[1, 1], 2);
        arr.set(&[2, 2], 3);

        assert_eq!(arr.get(&[0, 0]), 1);
        assert_eq!(arr.get(&[1, 1]), 2);
        assert_eq!(arr.get(&[2, 2]), 3);
    }

    #[test]
    fn size_matches_shape_product() {
        let arr: NdArray<i32> = NdArray::new((0..60).collect(), Shape::new(vec![3, 4, 5]));

        assert_eq!(arr.size(), 60);
    }

    #[test]
    fn strides_accessor_matches_stride_for() {
        let arr: NdArray<i32> = NdArray::new((0..24).collect(), Shape::new(vec![2, 3, 4]));

        assert_eq!(arr.strides().stride_for(0), arr.stride_for(0));

        assert_eq!(arr.strides().stride_for(1), arr.stride_for(1));

        assert_eq!(arr.strides().stride_for(2), arr.stride_for(2));
    }
}
