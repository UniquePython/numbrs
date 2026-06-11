use crate::{buffer::RawBuffer, layout::index_to_offset, shape::Shape, strides::Strides};

use std::cell::RefCell;
use std::rc::Rc;

pub struct AxisSlice {
    pub start: usize,
    pub end: usize,
    pub step: usize,
}

pub struct NdArray<T> {
    data: Rc<RefCell<RawBuffer<T>>>,
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
            data: Rc::new(RefCell::new(RawBuffer::new(data))),
            offset: 0,
            shape,
            strides,
        }
    }

    fn from_shared(
        data: Rc<RefCell<RawBuffer<T>>>,
        offset: usize,
        shape: Shape,
        strides: Strides,
    ) -> Self {
        Self {
            data,
            offset,
            shape,
            strides,
        }
    }

    pub fn row(&self, index: usize) -> NdArray<T> {
        if self.rank() != 2 {
            panic!("row() requires a 2D array, found rank {}", self.rank());
        }

        if index >= self.size_for(0) {
            panic!(
                "Row index {} out of bounds for axis 0 with size {}",
                index,
                self.size_for(0)
            );
        }

        let offset: usize = self.offset + index * self.stride_for(0);

        NdArray::from_shared(
            Rc::clone(&self.data),
            offset,
            Shape::new(vec![self.size_for(1)]),
            Strides::new(vec![self.stride_for(1)]),
        )
    }

    pub fn col(&self, index: usize) -> NdArray<T> {
        if self.rank() != 2 {
            panic!("col() requires a 2D array, found rank {}", self.rank());
        }

        if index >= self.size_for(1) {
            panic!(
                "Column index {} out of bounds for axis 1 with size {}",
                index,
                self.size_for(1)
            );
        }

        let offset: usize = self.offset + index * self.stride_for(1);

        NdArray::from_shared(
            Rc::clone(&self.data),
            offset,
            Shape::new(vec![self.size_for(0)]),
            Strides::new(vec![self.stride_for(0)]),
        )
    }

    pub fn slice(&self, slices: &[AxisSlice]) -> NdArray<T> {
        if slices.len() != self.rank() {
            panic!(
                "Expected {} slices for rank {}, found {}",
                self.rank(),
                self.rank(),
                slices.len()
            );
        }

        let mut offset: usize = self.offset;
        let mut shape: Vec<usize> = Vec::with_capacity(self.rank());
        let mut strides: Vec<usize> = Vec::with_capacity(self.rank());

        for (axis, slice) in slices.iter().enumerate() {
            let axis_size: usize = self.size_for(axis);

            if slice.step == 0 {
                panic!("Slice step must be at least 1");
            }

            if slice.start >= slice.end {
                panic!(
                    "Slice start ({}) must be less than end ({})",
                    slice.start, slice.end
                );
            }

            if slice.end > axis_size {
                panic!(
                    "Slice end ({}) exceeds axis {} size ({})",
                    slice.end, axis, axis_size
                );
            }

            let stride: usize = self.stride_for(axis);

            offset += slice.start * stride;

            let new_size: usize = (slice.end - slice.start + slice.step - 1) / slice.step;

            shape.push(new_size);
            strides.push(stride * slice.step);
        }

        NdArray::from_shared(
            Rc::clone(&self.data),
            offset,
            Shape::new(shape),
            Strides::new(strides),
        )
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

        self.data.borrow().get(self.offset + local_offset).clone()
    }

    pub fn set(&mut self, indices: &[usize], value: T) {
        let local_offset: usize = index_to_offset(indices, &self.strides, &self.shape);

        self.data
            .borrow_mut()
            .set(self.offset + local_offset, value);
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
    fn row_returns_correct_shape() {
        let arr: NdArray<i32> = NdArray::new(vec![1, 2, 3, 4, 5, 6], Shape::new(vec![2, 3]));

        let row: NdArray<i32> = arr.row(1);

        assert_eq!(row.rank(), 1);
        assert_eq!(row.shape(), &Shape::new(vec![3]));
    }

    #[test]
    fn row_returns_correct_values() {
        let arr: NdArray<i32> = NdArray::new(vec![1, 2, 3, 4, 5, 6], Shape::new(vec![2, 3]));

        let row: NdArray<i32> = arr.row(1);

        assert_eq!(row.get(&[0]), 4);
        assert_eq!(row.get(&[1]), 5);
        assert_eq!(row.get(&[2]), 6);
    }

    #[test]
    fn row_is_a_view() {
        let arr: NdArray<i32> = NdArray::new(vec![1, 2, 3, 4, 5, 6], Shape::new(vec![2, 3]));

        let mut row: NdArray<i32> = arr.row(1);

        row.set(&[1], 99);

        assert_eq!(arr.get(&[1, 1]), 99);
    }

    #[test]
    fn col_returns_correct_shape() {
        let arr: NdArray<i32> = NdArray::new(vec![1, 2, 3, 4, 5, 6], Shape::new(vec![2, 3]));

        let col: NdArray<i32> = arr.col(1);

        assert_eq!(col.rank(), 1);
        assert_eq!(col.shape(), &Shape::new(vec![2]));
    }

    #[test]
    fn col_returns_correct_values() {
        let arr: NdArray<i32> = NdArray::new(vec![1, 2, 3, 4, 5, 6], Shape::new(vec![2, 3]));

        let col: NdArray<i32> = arr.col(1);

        assert_eq!(col.get(&[0]), 2);
        assert_eq!(col.get(&[1]), 5);
    }

    #[test]
    fn col_is_a_view() {
        let arr: NdArray<i32> = NdArray::new(vec![1, 2, 3, 4, 5, 6], Shape::new(vec![2, 3]));

        let mut col: NdArray<i32> = arr.col(1);

        col.set(&[0], 99);

        assert_eq!(arr.get(&[0, 1]), 99);
    }

    #[test]
    fn slice_contiguous_on_one_axis() {
        let arr: NdArray<i32> = NdArray::new(vec![1, 2, 3, 4, 5, 6], Shape::new(vec![2, 3]));

        let slice: NdArray<i32> = arr.slice(&[
            AxisSlice {
                start: 0,
                end: 2,
                step: 1,
            },
            AxisSlice {
                start: 1,
                end: 3,
                step: 1,
            },
        ]);

        assert_eq!(slice.shape(), &Shape::new(vec![2, 2]));

        assert_eq!(slice.get(&[0, 0]), 2);
        assert_eq!(slice.get(&[0, 1]), 3);
        assert_eq!(slice.get(&[1, 0]), 5);
        assert_eq!(slice.get(&[1, 1]), 6);
    }

    #[test]
    fn slice_on_both_axes() {
        let arr: NdArray<i32> = NdArray::new((0..16).collect(), Shape::new(vec![4, 4]));

        let slice: NdArray<i32> = arr.slice(&[
            AxisSlice {
                start: 1,
                end: 3,
                step: 1,
            },
            AxisSlice {
                start: 1,
                end: 4,
                step: 1,
            },
        ]);

        assert_eq!(slice.shape(), &Shape::new(vec![2, 3]));

        assert_eq!(slice.get(&[0, 0]), 5);
        assert_eq!(slice.get(&[0, 1]), 6);
        assert_eq!(slice.get(&[0, 2]), 7);

        assert_eq!(slice.get(&[1, 0]), 9);
        assert_eq!(slice.get(&[1, 1]), 10);
        assert_eq!(slice.get(&[1, 2]), 11);
    }

    #[test]
    fn slice_with_step() {
        let arr: NdArray<i32> = NdArray::new((0..16).collect(), Shape::new(vec![4, 4]));

        let slice: NdArray<i32> = arr.slice(&[
            AxisSlice {
                start: 0,
                end: 4,
                step: 2,
            },
            AxisSlice {
                start: 0,
                end: 4,
                step: 2,
            },
        ]);

        assert_eq!(slice.shape(), &Shape::new(vec![2, 2]));

        assert_eq!(slice.get(&[0, 0]), 0);
        assert_eq!(slice.get(&[0, 1]), 2);
        assert_eq!(slice.get(&[1, 0]), 8);
        assert_eq!(slice.get(&[1, 1]), 10);
    }

    #[test]
    fn slice_updates_strides_for_steps() {
        let arr: NdArray<i32> = NdArray::new((0..16).collect(), Shape::new(vec![4, 4]));

        let slice: NdArray<i32> = arr.slice(&[
            AxisSlice {
                start: 0,
                end: 4,
                step: 2,
            },
            AxisSlice {
                start: 0,
                end: 4,
                step: 2,
            },
        ]);

        assert_eq!(slice.stride_for(0), 8);
        assert_eq!(slice.stride_for(1), 2);
    }

    #[test]
    fn slice_is_a_view() {
        let arr: NdArray<i32> = NdArray::new((0..16).collect(), Shape::new(vec![4, 4]));

        let mut slice: NdArray<i32> = arr.slice(&[
            AxisSlice {
                start: 1,
                end: 3,
                step: 1,
            },
            AxisSlice {
                start: 1,
                end: 3,
                step: 1,
            },
        ]);

        slice.set(&[0, 0], 99);

        assert_eq!(arr.get(&[1, 1]), 99);
    }

    #[test]
    fn slice_step_changes_access_pattern() {
        let arr: NdArray<i32> = NdArray::new((0..10).collect(), Shape::new(vec![10]));

        let slice: NdArray<i32> = arr.slice(&[AxisSlice {
            start: 1,
            end: 10,
            step: 3,
        }]);

        assert_eq!(slice.shape(), &Shape::new(vec![3]));

        assert_eq!(slice.get(&[0]), 1);
        assert_eq!(slice.get(&[1]), 4);
        assert_eq!(slice.get(&[2]), 7);
    }

    #[test]
    #[should_panic]
    fn slice_panics_when_slice_count_does_not_match_rank() {
        let arr: NdArray<i32> = NdArray::new(vec![1, 2, 3, 4], Shape::new(vec![2, 2]));

        arr.slice(&[AxisSlice {
            start: 0,
            end: 2,
            step: 1,
        }]);
    }

    #[test]
    #[should_panic]
    fn slice_panics_when_start_equals_end() {
        let arr: NdArray<i32> = NdArray::new(vec![1, 2, 3, 4], Shape::new(vec![2, 2]));

        arr.slice(&[
            AxisSlice {
                start: 0,
                end: 0,
                step: 1,
            },
            AxisSlice {
                start: 0,
                end: 2,
                step: 1,
            },
        ]);
    }

    #[test]
    #[should_panic]
    fn slice_panics_when_start_greater_than_end() {
        let arr: NdArray<i32> = NdArray::new(vec![1, 2, 3, 4], Shape::new(vec![2, 2]));

        arr.slice(&[
            AxisSlice {
                start: 1,
                end: 0,
                step: 1,
            },
            AxisSlice {
                start: 0,
                end: 2,
                step: 1,
            },
        ]);
    }

    #[test]
    #[should_panic]
    fn slice_panics_when_end_exceeds_axis_size() {
        let arr: NdArray<i32> = NdArray::new(vec![1, 2, 3, 4], Shape::new(vec![2, 2]));

        arr.slice(&[
            AxisSlice {
                start: 0,
                end: 3,
                step: 1,
            },
            AxisSlice {
                start: 0,
                end: 2,
                step: 1,
            },
        ]);
    }

    #[test]
    #[should_panic]
    fn slice_panics_when_step_is_zero() {
        let arr: NdArray<i32> = NdArray::new(vec![1, 2, 3, 4], Shape::new(vec![2, 2]));

        arr.slice(&[
            AxisSlice {
                start: 0,
                end: 2,
                step: 0,
            },
            AxisSlice {
                start: 0,
                end: 2,
                step: 1,
            },
        ]);
    }

    #[test]
    fn slice_applies_offset_correctly() {
        let arr: NdArray<i32> = NdArray::new((0..16).collect(), Shape::new(vec![4, 4]));

        let slice: NdArray<i32> = arr.slice(&[
            AxisSlice {
                start: 2,
                end: 4,
                step: 1,
            },
            AxisSlice {
                start: 1,
                end: 3,
                step: 1,
            },
        ]);

        assert_eq!(slice.get(&[0, 0]), 9);
        assert_eq!(slice.get(&[0, 1]), 10);
        assert_eq!(slice.get(&[1, 0]), 13);
        assert_eq!(slice.get(&[1, 1]), 14);
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
    #[should_panic]
    fn row_panics_on_1d_array() {
        let arr: NdArray<i32> = NdArray::new(vec![1, 2, 3], Shape::new(vec![3]));

        arr.row(0);
    }

    #[test]
    #[should_panic]
    fn col_panics_on_1d_array() {
        let arr: NdArray<i32> = NdArray::new(vec![1, 2, 3], Shape::new(vec![3]));

        arr.col(0);
    }

    #[test]
    #[should_panic]
    fn row_panics_when_index_out_of_bounds() {
        let arr: NdArray<i32> = NdArray::new(vec![1, 2, 3, 4], Shape::new(vec![2, 2]));

        arr.row(2);
    }

    #[test]
    #[should_panic]
    fn col_panics_when_index_out_of_bounds() {
        let arr: NdArray<i32> = NdArray::new(vec![1, 2, 3, 4], Shape::new(vec![2, 2]));

        arr.col(2);
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
