use std::cmp::Ordering;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use lazy_static::lazy_static;
use pyo3::prelude::*;

// String interning for sort field
lazy_static! {
    static ref STRING_INTERNER: Mutex<HashMap<String, Arc<String>>> = Mutex::new(HashMap::new());
}

// Helper function to intern strings
fn intern_string(s: String) -> Arc<String> {
    let mut interner = STRING_INTERNER.lock().unwrap();
    if let Some(interned) = interner.get(&s) {
        interned.clone()
    } else {
        let arc = Arc::new(s.clone());
        interner.insert(s, arc.clone());
        arc
    }
}

#[pyclass]
#[derive(Clone)]
pub struct Segment {
    #[pyo3(get, set)]
    pub start: u64,
    #[pyo3(get, set)]
    pub end: u64,
    // Internal field uses Arc<String> for efficient comparison
    sort: Arc<String>,
}

#[pymethods]
impl Segment {
    #[new]
    pub fn new(start: u64, end: u64, sort: String) -> Self {
        Segment {
            start,
            end,
            sort: intern_string(sort),
        }
    }

    pub fn copy(&self) -> Self {
        self.clone()
    }

    #[getter]
    pub fn size(&self) -> u64 {
        self.end - self.start
    }

    // Getter for sort to maintain Python interface
    #[getter]
    pub fn sort(&self) -> String {
        self.sort.to_string()
    }

    // Setter for sort to maintain Python interface
    #[setter]
    pub fn set_sort(&mut self, value: String) {
        self.sort = intern_string(value);
    }

    fn __repr__(&self) -> String {
        format!("[{:#x}-{:#x}, {}]", self.start, self.end, self.sort)
    }
}

#[derive(Clone, Default)]
#[pyclass]
pub struct SegmentList {
    list: Vec<Segment>,
    bytes_occupied: u64,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Direction {
    Forward,
    Backward,
}

impl SegmentList {
    // Private helper methods for segment management

    fn insert_and_merge(&mut self, address: u64, size: u64, sort: String, idx: usize) {
        // Sanity check
        if idx > 0 && address + size <= self.list[idx - 1].start {
            // There is a bug, since list[idx] must be the closest one that is less than the current segment
            // Let's fix it by recursively calling with idx - 1
            self.insert_and_merge(address, size, sort, idx - 1);
            return;
        }

        // Insert the block first
        // The new block might be overlapping with other blocks. insert_and_merge_core will fix the overlapping.
        let end = address + size;
        if idx == self.list.len() {
            self.list.push(Segment::new(address, end, sort));
        } else {
            self.list.insert(idx, Segment::new(address, end, sort));
        }
        // Apparently bytes_occupied will be wrong if the new block overlaps with any existing block.
        // We will fix it later
        self.bytes_occupied += &size; // Use reference instead of clone

        // Search forward to merge blocks if necessary
        let mut pos = idx;
        while pos < self.list.len() {
            let (merged, new_pos, bytes_change) =
                self.insert_and_merge_core(pos, Direction::Forward);
            if !merged {
                break;
            }
            pos = new_pos;
            if bytes_change != 0 {
                self.bytes_occupied += &bytes_change;
            }
        }

        // Search backward to merge blocks if necessary
        let mut pos = idx;
        while pos > 0 {
            let (merged, new_pos, bytes_change) =
                self.insert_and_merge_core(pos, Direction::Backward);
            if !merged {
                break;
            }
            pos = new_pos;
            if bytes_change != 0 {
                self.bytes_occupied += &bytes_change;
            }
        }
    }

    fn insert_and_merge_core(&mut self, pos: usize, direction: Direction) -> (bool, usize, u64) {
        let mut bytes_changed = 0;
        let mut merged = false;
        let mut new_pos = pos;

        // Determine previous_segment and segment based on direction
        let (previous_segment_pos, segment_pos) = match direction {
            Direction::Forward => {
                if pos == self.list.len() - 1 {
                    return (false, pos, 0);
                }
                (pos, pos + 1)
            }
            Direction::Backward => {
                if pos == 0 {
                    return (false, pos, 0);
                }
                (pos - 1, pos)
            }
        };

        // Get references to the segments
        let previous_segment = &self.list[previous_segment_pos].clone();
        let segment = &self.list[segment_pos].clone();

        if segment.start <= previous_segment.end {
            // We should always have new_start + new_size >= segment.start

            if segment.sort == previous_segment.sort {
                // They are of the same sort - we should merge them!
                let segment_size = segment.end - segment.start;
                let new_end = if segment.end > previous_segment.end {
                    segment.end
                } else {
                    previous_segment.end
                };
                let new_start = if segment.start < previous_segment.start {
                    segment.start
                } else {
                    previous_segment.start
                };
                let new_size = new_end - new_start;

                // Update segment_pos with the merged segment
                self.list[segment_pos] = Segment {
                    start: new_start,
                    end: new_end,
                    sort: segment.sort.clone(),
                };

                // Remove previous_segment_pos
                self.list.remove(previous_segment_pos);

                // Calculate bytes_changed
                let previous_size = previous_segment.end - previous_segment.start;
                let old_size = segment_size + previous_size;

                // Calculate bytes_changed safely
                match new_size.cmp(&old_size) {
                    Ordering::Greater => {
                        bytes_changed = new_size - old_size;
                    }
                    Ordering::Less => {
                        // If old_size > new_size, we need to subtract from bytes_occupied
                        // We'll do this directly
                        self.bytes_occupied -= old_size - new_size;
                        // Return zero for bytes_changed since we've already updated bytes_occupied
                        bytes_changed = u64::from(0u8);
                    }
                    Ordering::Equal => {}
                }

                merged = true;
                new_pos = previous_segment_pos;
            } else {
                // Different sorts. It's a bit trickier.
                if segment.start == previous_segment.end {
                    // They are adjacent. Just don't merge.
                } else {
                    // They are overlapping. We will create one, two, or three different blocks based on how they are
                    // overlapping
                    let mut new_segments = Vec::with_capacity(3);

                    if segment.start < previous_segment.start {
                        new_segments.push(Segment {
                            start: segment.start,
                            end: previous_segment.start,
                            sort: segment.sort.clone(),
                        });

                        let sort = match direction {
                            Direction::Forward => previous_segment.sort.clone(),
                            Direction::Backward => segment.sort.clone(),
                        };

                        new_segments.push(Segment {
                            start: previous_segment.start,
                            end: previous_segment.end,
                            sort,
                        });

                        match segment.end.cmp(&previous_segment.end) {
                            Ordering::Less => {
                                new_segments.push(Segment {
                                    start: segment.end,
                                    end: previous_segment.end,
                                    sort: previous_segment.sort.clone(),
                                });
                            }
                            Ordering::Equal => {}
                            Ordering::Greater => {
                                new_segments.push(Segment {
                                    start: previous_segment.end,
                                    end: segment.end,
                                    sort: segment.sort.clone(),
                                });
                            }
                        }
                    } else {
                        // segment.start >= previous_segment.start
                        if segment.start > previous_segment.start {
                            new_segments.push(Segment {
                                start: previous_segment.start,
                                end: segment.start,
                                sort: previous_segment.sort.clone(),
                            });
                        }

                        let sort = match direction {
                            Direction::Forward => previous_segment.sort.clone(),
                            Direction::Backward => segment.sort.clone(),
                        };

                        match segment.end.cmp(&previous_segment.end) {
                            Ordering::Less => {
                                new_segments.push(Segment {
                                    start: segment.start,
                                    end: segment.end,
                                    sort: sort.clone(),
                                });
                                new_segments.push(Segment {
                                    start: segment.end,
                                    end: previous_segment.end,
                                    sort: previous_segment.sort.clone(),
                                });
                            }
                            Ordering::Equal => {
                                new_segments.push(Segment {
                                    start: segment.start,
                                    end: segment.end,
                                    sort: sort.clone(),
                                });
                            }
                            Ordering::Greater => {
                                new_segments.push(Segment {
                                    start: segment.start,
                                    end: previous_segment.end,
                                    sort: sort.clone(),
                                });
                                new_segments.push(Segment {
                                    start: previous_segment.end,
                                    end: segment.end,
                                    sort: segment.sort.clone(),
                                });
                            }
                        }
                    }

                    // Merge segments in new_segments array if they are of the same sort
                    let mut i = 0;
                    while new_segments.len() > 1 && i < new_segments.len() - 1 {
                        let s0 = &new_segments[i];
                        let s1 = &new_segments[i + 1];

                        if s0.sort == s1.sort {
                            let merged_segment = Segment {
                                start: s0.start,
                                end: s1.end,
                                sort: s0.sort.clone(),
                            };

                            // Replace s0 with merged_segment and remove s1
                            new_segments[i] = merged_segment;
                            new_segments.remove(i + 1);
                        } else {
                            i += 1;
                        }
                    }

                    // Calculate old_size and new_size
                    let mut old_size = 0;
                    for i in previous_segment_pos..=segment_pos {
                        old_size += self.list[i].end - self.list[i].start;
                    }

                    let mut new_size = 0;
                    for seg in &new_segments {
                        new_size += seg.end - seg.start;
                    }

                    // Calculate bytes_changed safely
                    match new_size.cmp(&old_size) {
                        Ordering::Greater => {
                            bytes_changed = new_size - old_size;
                        }
                        Ordering::Less => {
                            // If old_size > new_size, we need to subtract from bytes_occupied
                            // We'll do this directly
                            self.bytes_occupied -= old_size - new_size;
                            // Return zero for bytes_changed since we've already updated bytes_occupied
                            bytes_changed = 0;
                        }
                        Ordering::Equal => {}
                    }

                    // Put new segments into self.list
                    match new_segments.len() {
                        1 => {
                            self.list[previous_segment_pos] = new_segments[0].clone();
                            self.list.remove(segment_pos);
                        }
                        2 => {
                            self.list[previous_segment_pos] = new_segments[0].clone();
                            self.list[segment_pos] = new_segments[1].clone();
                        }
                        3 => {
                            self.list[previous_segment_pos] = new_segments[0].clone();
                            self.list[segment_pos] = new_segments[1].clone();
                            self.list.insert(segment_pos + 1, new_segments[2].clone());
                        }
                        _ => {
                            // This does not happen for now, but may happen when the above logic changes
                            self.list.splice(
                                previous_segment_pos..=segment_pos,
                                new_segments.iter().cloned(),
                            );
                        }
                    }

                    merged = true;

                    if direction == Direction::Forward {
                        new_pos = previous_segment_pos + new_segments.len() - 1;
                    } else {
                        new_pos = previous_segment_pos;
                    }
                }
            }
        }

        (merged, new_pos, bytes_changed)
    }

    fn remove(&mut self, init_address: u64, init_size: u64, init_idx: usize) {
        let mut address = init_address;
        let mut size = init_size;
        let mut idx = init_idx;

        while idx < self.list.len() {
            let segment = self.list[idx].clone();

            if segment.start <= address {
                match segment.end.cmp(&(address + size)) {
                    Ordering::Less => {
                        // |---segment---|
                        //      |---address + size---|
                        // shrink segment
                        let old_size = segment.end - segment.start;
                        self.list[idx].end = address;
                        let new_size = self.list[idx].end - self.list[idx].start;
                        self.bytes_occupied -= old_size - new_size;

                        if (self.list[idx].end - self.list[idx].start) == 0 {
                            // remove the segment
                            self.list.remove(idx);
                        }

                        // adjust address
                        let new_address = segment.end;
                        // adjust size
                        size = address + size - new_address;
                        address = new_address;
                        // update idx
                        idx = self.search(address);
                    }
                    Ordering::Equal | Ordering::Greater => {
                        // |--------segment--------|
                        //    |--address + size--|
                        // break segment
                        let old_size = segment.end - segment.start;

                        let seg0 = Segment {
                            start: segment.start,
                            end: address,
                            sort: segment.sort.clone(),
                        };

                        let seg1 = Segment {
                            start: address + size,
                            end: segment.end,
                            sort: segment.sort,
                        };

                        // Calculate new sizes
                        let seg0_size = seg0.end - seg0.start;
                        let seg1_size = seg1.end - seg1.start;

                        // Update bytes_occupied
                        self.bytes_occupied -= old_size - seg0_size - seg1_size;

                        // remove the current segment
                        self.list.remove(idx);

                        // Insert new segments if they have size > 0
                        if (seg1.end - seg1.start) != 0 {
                            self.list.insert(idx, seg1);
                        }

                        if (seg0.end - seg0.start) != 0 {
                            self.list.insert(idx, seg0);
                        }

                        // done
                        break;
                    }
                }
            } else {
                // if segment.start > address
                match (address + size).cmp(&segment.start) {
                    Ordering::Less | Ordering::Equal => {
                        //                      |--- segment ---|
                        // |-- address + size --|
                        // no overlap
                        break;
                    }
                    Ordering::Greater => {
                        // Overlap exists
                        match (address + size).cmp(&segment.end) {
                            Ordering::Less | Ordering::Equal => {
                                //            |---- segment ----|
                                // |-- address + size --|
                                //
                                // update the start of the segment
                                let old_size = segment.end - segment.start;
                                self.list[idx].start = address + size;
                                let new_size = self.list[idx].end - self.list[idx].start;
                                self.bytes_occupied -= old_size - new_size;

                                if (self.list[idx].end - self.list[idx].start) == 0 {
                                    // remove the segment
                                    self.list.remove(idx);
                                }

                                break;
                            }
                            Ordering::Greater => {
                                //            |---- segment ----|
                                // |--------- address + size ----------|
                                let old_size = segment.end - segment.start;
                                self.bytes_occupied -= old_size;

                                self.list.remove(idx); // remove the segment

                                let new_address = segment.end;
                                size = address + size - new_address;
                                address = new_address;
                                idx = self.search(address);
                            }
                        }
                    }
                }
            }
        }
    }
}

#[pymethods]
impl SegmentList {
    #[new]
    pub fn new() -> Self {
        Default::default()
    }

    pub fn __len__(&self) -> usize {
        self.list.len()
    }

    pub fn __getitem__(&self, idx: usize) -> PyResult<Segment> {
        self.list.get(idx).cloned().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!("Index {} out of range", idx))
        })
    }

    #[getter]
    pub fn occupied_size(&self) -> u64 {
        // Clone only at the boundary
        self.bytes_occupied
    }

    #[getter]
    pub fn has_blocks(&self) -> bool {
        !self.list.is_empty()
    }

    pub fn search(&self, addr: u64) -> usize {
        // Match Python implementation behavior but with more efficient comparison
        let mut off = self
            .list
            .binary_search_by(|seg| {
                if seg.start > addr {
                    Ordering::Greater
                } else {
                    Ordering::Less
                }
            })
            .unwrap_or_else(|x| x);

        // Check if we need to adjust the offset
        if off > 0 && off <= self.list.len() && self.list[off - 1].end > addr {
            off -= 1;
        }

        off
    }

    pub fn next_free_pos(&self, address: u64) -> u64 {
        // Match Python implementation behavior
        let idx = self.search(address);

        if idx < self.list.len() && self.list[idx].start <= address && address < self.list[idx].end
        {
            // Address is inside a segment
            let mut i = idx;
            while i + 1 < self.list.len() && self.list[i].end == self.list[i + 1].start {
                i += 1;
            }

            if i == self.list.len() {
                return self.list[self.list.len() - 1].end;
            }

            return self.list[i].end;
        }

        // Address is not inside any segment
        address
    }

    #[pyo3(signature = (address, sorts, max_distance = None))]
    pub fn next_pos_with_sort_not_in(
        &self,
        address: u64,
        sorts: Vec<String>,
        max_distance: Option<u64>,
    ) -> Option<u64> {
        // Match Python implementation behavior
        let list_length = self.list.len();
        let idx = self.search(address);

        if idx < list_length {
            // Check max_distance constraint with the block
            let block = &self.list[idx];
            if let Some(ref max_dist) = max_distance {
                if address + max_dist < block.start {
                    return None;
                }
            }

            // Check if address is inside the current block
            if block.start <= address && address < block.end {
                // If the current block's sort is not in the excluded sorts, return the address
                if !sorts.contains(&block.sort) {
                    return Some(address);
                }

                // Otherwise, search forward for a block with a different sort
                let mut i = idx + 1;
                while i < list_length {
                    let seg = &self.list[i];

                    // Check max_distance constraint
                    if let Some(ref max_dist) = max_distance {
                        if address + max_dist < seg.start {
                            return None;
                        }
                    }

                    // If this segment's sort is not in the excluded sorts, return its start
                    if !sorts.contains(&seg.sort) {
                        return Some(seg.start);
                    }

                    i += 1;
                }

                return None;
            } else {
                // Address is not inside any block, start searching from the current index
                let mut i = idx;
                while i < list_length {
                    let seg = &self.list[i];

                    // Check max_distance constraint
                    if let Some(ref max_dist) = max_distance {
                        if address + max_dist < seg.start {
                            return None;
                        }
                    }

                    // If this segment's sort is not in the excluded sorts, return its start
                    if !sorts.contains(&seg.sort) {
                        return Some(seg.start);
                    }

                    i += 1;
                }

                return None;
            }
        }

        None
    }

    pub fn is_occupied(&self, address: u64) -> bool {
        // Match Python implementation behavior
        let idx = self.search(address);

        if idx < self.list.len() && self.list[idx].start <= address && address < self.list[idx].end
        {
            return true;
        }

        // Check if the address is in the previous segment
        if idx > 0 && address < self.list[idx - 1].end {
            return true;
        }

        false
    }

    pub fn occupied_by(&self, address: u64) -> Option<(u64, u64, String)> {
        // Match Python implementation behavior
        let idx = self.search(address);

        if idx < self.list.len() && self.list[idx].start <= address && address < self.list[idx].end
        {
            let block = &self.list[idx];
            return Some((
                block.start,
                (block.end - block.start),
                block.sort.to_string(),
            ));
        }

        // Check if the address is in the previous segment
        if idx > 0 && address < self.list[idx - 1].end {
            let block = &self.list[idx - 1];
            return Some((
                block.start,
                (block.end - block.start),
                block.sort.to_string(),
            ));
        }

        None
    }

    pub fn occupy(&mut self, address: u64, size: u64, sort: String) {
        if size == 0 {
            return;
        }

        if self.list.is_empty() {
            let end = address + size;
            self.list.push(Segment::new(address, end, sort));
            self.bytes_occupied += &size; // Use reference instead of clone
            return;
        }

        // Find adjacent element in our list
        let idx = self.search(address);

        // Insert and merge the new segment
        self.insert_and_merge(address, size, sort, idx);
    }

    pub fn release(&mut self, address: u64, size: u64) {
        if size == 0 {
            return;
        }

        if self.list.is_empty() {
            return;
        }

        let idx = self.search(address);
        if idx < self.list.len() {
            self.remove(address, size, idx);
        }
    }

    pub fn copy(&self) -> SegmentList {
        SegmentList {
            list: self.list.clone(),
            bytes_occupied: self.bytes_occupied,
        }
    }
}

#[pymodule]
fn segmentlist(m: Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Segment>()?;
    m.add_class::<SegmentList>()?;
    Ok(())
}
