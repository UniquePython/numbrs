use std::cell::RefCell;
use std::rc::Rc;

pub struct RawBuffer<T> {
    data: Rc<RefCell<Vec<T>>>,
    offset: usize,
    len: usize,
}

impl<T> RawBuffer<T> {
    pub fn new(data: Vec<T>) -> Self {
        let len: usize = data.len();
        Self {
            data: Rc::new(RefCell::new(data)),
            offset: 0,
            len,
        }
    }

    pub fn filled(value: T, len: usize) -> Self
    where
        T: Clone,
    {
        Self {
            data: Rc::new(RefCell::new(vec![value; len])),
            offset: 0,
            len,
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn bounds_check(&self, index: usize) {
        if index >= self.len() {
            panic!(
                "Index {index} is out of bounds: [0, {len})",
                len = self.len()
            );
        }
    }

    pub fn get(&self, index: usize) -> T
    where
        T: Clone,
    {
        self.bounds_check(index);
        self.data.borrow()[self.offset + index].clone()
    }

    pub fn set(&mut self, index: usize, value: T) {
        self.bounds_check(index);
        self.data.borrow_mut()[self.offset + index] = value;
    }

    pub fn slice(&self, start: usize, len: usize) -> Self {
        if start + len > self.len() {
            panic!(
                "Index {} is out of bounds: [0, {len})",
                start + len,
                len = self.len()
            );
        }

        Self {
            data: self.data.clone(),
            offset: self.offset + start,
            len,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_stores_values_correctly() {
        let rb: RawBuffer<i32> = RawBuffer::new(vec![10, 20, 30]);
        assert_eq!(rb.get(0), 10);
        assert_eq!(rb.get(1), 20);
        assert_eq!(rb.get(2), 30);
    }

    #[test]
    fn filled_creates_repeated_values() {
        let rb: RawBuffer<i32> = RawBuffer::filled(7, 5);
        assert_eq!(rb.len(), 5);

        for i in 0..5 {
            assert_eq!(rb.get(i), 7);
        }
    }

    #[test]
    fn len_and_is_empty_work() {
        let rb: RawBuffer<i32> = RawBuffer::new(vec![1, 2, 3]);
        assert_eq!(rb.len(), 3);
        assert!(!rb.is_empty());

        let empty: RawBuffer<i32> = RawBuffer::new(vec![]);
        assert_eq!(empty.len(), 0);
        assert!(empty.is_empty());
    }

    #[test]
    fn set_updates_values() {
        let mut rb: RawBuffer<i32> = RawBuffer::new(vec![1, 2, 3]);
        rb.set(1, 99);
        assert_eq!(rb.get(1), 99);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn get_panics_on_out_of_bounds() {
        let rb: RawBuffer<i32> = RawBuffer::new(vec![1, 2, 3]);
        rb.get(3);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn set_panics_on_out_of_bounds() {
        let mut rb: RawBuffer<i32> = RawBuffer::new(vec![1, 2, 3]);
        rb.set(10, 99);
    }

    #[test]
    fn slice_returns_correct_view() {
        let rb: RawBuffer<i32> = RawBuffer::new(vec![10, 20, 30, 40, 50]);

        let slice: RawBuffer<i32> = rb.slice(1, 3);

        assert_eq!(slice.len(), 3);
        assert_eq!(slice.get(0), 20);
        assert_eq!(slice.get(1), 30);
        assert_eq!(slice.get(2), 40);
    }

    #[test]
    fn slice_can_cover_entire_buffer() {
        let rb: RawBuffer<i32> = RawBuffer::new(vec![1, 2, 3]);

        let slice: RawBuffer<i32> = rb.slice(0, 3);

        assert_eq!(slice.len(), 3);
        assert_eq!(slice.get(0), 1);
        assert_eq!(slice.get(1), 2);
        assert_eq!(slice.get(2), 3);
    }

    #[test]
    fn slice_can_be_empty() {
        let rb: RawBuffer<i32> = RawBuffer::new(vec![1, 2, 3]);

        let slice: RawBuffer<i32> = rb.slice(1, 0);

        assert_eq!(slice.len(), 0);
        assert!(slice.is_empty());
    }

    #[test]
    fn nested_slices_work_correctly() {
        let rb: RawBuffer<i32> = RawBuffer::new(vec![10, 20, 30, 40, 50]);

        let slice1: RawBuffer<i32> = rb.slice(1, 4); // [20, 30, 40, 50]
        let slice2: RawBuffer<i32> = slice1.slice(1, 2); // [30, 40]

        assert_eq!(slice2.len(), 2);
        assert_eq!(slice2.get(0), 30);
        assert_eq!(slice2.get(1), 40);
    }

    #[test]
    fn modifying_slice_affects_original_buffer() {
        let rb: RawBuffer<i32> = RawBuffer::new(vec![10, 20, 30]);

        {
            let mut slice: RawBuffer<i32> = rb.slice(1, 2);
            slice.set(0, 99);
        }

        assert_eq!(rb.get(0), 10);
        assert_eq!(rb.get(1), 99);
        assert_eq!(rb.get(2), 30);
    }

    #[test]
    fn modifying_original_affects_slice() {
        let mut rb: RawBuffer<i32> = RawBuffer::new(vec![10, 20, 30]);

        let slice: RawBuffer<i32> = rb.slice(1, 2);

        rb.set(2, 99);

        assert_eq!(slice.get(0), 20);
        assert_eq!(slice.get(1), 99);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn slice_panics_when_range_exceeds_buffer() {
        let rb: RawBuffer<i32> = RawBuffer::new(vec![1, 2, 3]);

        rb.slice(2, 2);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn slice_panics_when_start_is_out_of_bounds() {
        let rb: RawBuffer<i32> = RawBuffer::new(vec![1, 2, 3]);

        rb.slice(4, 0);
    }
}
