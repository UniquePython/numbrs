pub struct RawBuffer<T> {
    data: Vec<T>,
}

impl<T> RawBuffer<T> {
    pub fn new(data: Vec<T>) -> Self {
        Self { data }
    }

    pub fn filled(value: T, len: usize) -> Self
    where
        T: Clone,
    {
        Self {
            data: vec![value; len],
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    fn bounds_check(&self, index: usize) {
        if index >= self.len() {
            panic!(
                "Index {index} is out of bounds: [0, {len})",
                len = self.len()
            );
        }
    }

    pub fn get(&self, index: usize) -> &T {
        self.bounds_check(index);
        &self.data[index]
    }

    pub fn get_mut(&mut self, index: usize) -> &mut T {
        self.bounds_check(index);
        &mut self.data[index]
    }

    pub fn set(&mut self, index: usize, value: T) {
        self.bounds_check(index);
        self.data[index] = value;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_stores_values_correctly() {
        let rb: RawBuffer<i32> = RawBuffer::new(vec![10, 20, 30]);
        assert_eq!(rb.get(0), &10);
        assert_eq!(rb.get(1), &20);
        assert_eq!(rb.get(2), &30);
    }

    #[test]
    fn filled_creates_repeated_values() {
        let rb: RawBuffer<i32> = RawBuffer::filled(7, 5);
        assert_eq!(rb.len(), 5);

        for i in 0..5 {
            assert_eq!(rb.get(i), &7);
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
        assert_eq!(rb.get(1), &99);
    }

    #[test]
    fn get_mut_allows_mutation() {
        let mut rb: RawBuffer<i32> = RawBuffer::new(vec![1, 2, 3]);
        *rb.get_mut(2) = 42;
        assert_eq!(rb.get(2), &42);
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
    #[should_panic(expected = "out of bounds")]
    fn get_mut_panics_on_out_of_bounds() {
        let mut rb: RawBuffer<i32> = RawBuffer::new(vec![1, 2, 3]);
        rb.get_mut(5);
    }
}
