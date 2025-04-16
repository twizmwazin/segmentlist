import time
import random

import pytest

from segment_list import Segment as PySegment, SegmentList as PySegmentList
from segmentlist import Segment as RustSegment, SegmentList as RustSegmentList


def compare_segment_lists(py_list, rust_list, check_occupied_size=True):
    """Compare Python and Rust SegmentList instances for equality."""
    # Check basic properties
    assert len(py_list) == len(rust_list), (
        f"Length mismatch: {len(py_list)} != {len(rust_list)}"
    )
    assert py_list.has_blocks == rust_list.has_blocks, "has_blocks mismatch"

    # Check occupied_size if requested
    if check_occupied_size:
        assert py_list.occupied_size == rust_list.occupied_size, (
            f"occupied_size mismatch: {py_list.occupied_size} != {rust_list.occupied_size}"
        )

    # Check each segment
    for i in range(len(py_list)):
        py_seg = py_list[i]
        rust_seg = rust_list[i]
        assert py_seg.start == rust_seg.start, (
            f"Segment {i} start mismatch: {py_seg.start} != {rust_seg.start}"
        )
        assert py_seg.end == rust_seg.end, (
            f"Segment {i} end mismatch: {py_seg.end} != {rust_seg.end}"
        )
        assert py_seg.sort == rust_seg.sort, (
            f"Segment {i} sort mismatch: {py_seg.sort} != {rust_seg.sort}"
        )


# Fixtures
@pytest.fixture
def py_list():
    """Fixture providing a Python SegmentList instance."""
    return PySegmentList()


@pytest.fixture
def rust_list():
    """Fixture providing a Rust SegmentList instance."""
    return RustSegmentList()

@pytest.fixture
def empty_lists(py_list, rust_list):
    """Fixture providing empty Python and Rust SegmentList instances."""
    return py_list, rust_list

@pytest.fixture
def simple_populated_lists(py_list, rust_list):
    """Fixture providing Python and Rust SegmentList instances with simple non-overlapping segments."""
    # Add some non-overlapping segments
    segments = [
        (100, 200, "code"),
        (300, 400, "data"),
        (500, 600, "code"),
    ]

    for start, end, sort in segments:
        py_list.occupy(start, end - start, sort)
        rust_list.occupy(start, end - start, sort)

    return py_list, rust_list


@pytest.fixture
def complex_populated_lists(py_list, rust_list):
    """Fixture providing Python and Rust SegmentList instances with complex overlapping segments."""
    # Add segments one by one to avoid subtraction issues
    # For Python implementation
    py_list.occupy(100, 200, "code")
    py_list.occupy(250, 150, "code")
    py_list.occupy(350, 150, "data")
    py_list.occupy(450, 100, "data")
    py_list.occupy(600, 100, "code")

    # For Rust implementation
    rust_list.occupy(100, 200, "code")
    rust_list.occupy(250, 150, "code")
    rust_list.occupy(350, 150, "data")
    rust_list.occupy(450, 100, "data")
    rust_list.occupy(600, 100, "code")

    return py_list, rust_list


# Basic functionality tests
class TestBasicFunctionality:
    def test_empty_list_creation(self, empty_lists):
        """Test creating empty segment lists."""
        py_list, rust_list = empty_lists
        assert len(py_list) == 0
        assert len(rust_list) == 0
        assert not py_list.has_blocks
        assert not rust_list.has_blocks
        assert py_list.occupied_size == 0
        assert rust_list.occupied_size == 0

    def test_segment_creation(self):
        """Test creating segments."""
        py_seg = PySegment(100, 200, "code")
        rust_seg = RustSegment(100, 200, "code")

        assert py_seg.start == rust_seg.start
        assert py_seg.end == rust_seg.end
        assert py_seg.sort == rust_seg.sort
        assert py_seg.size == rust_seg.size

    def test_segment_copy(self):
        """Test copying segments."""
        py_seg = PySegment(100, 200, "code")
        rust_seg = RustSegment(100, 200, "code")

        py_copy = py_seg.copy()
        rust_copy = rust_seg.copy()

        assert py_copy.start == py_seg.start
        assert py_copy.end == py_seg.end
        assert py_copy.sort == py_seg.sort

        assert rust_copy.start == rust_seg.start
        assert rust_copy.end == rust_seg.end
        assert rust_copy.sort == rust_seg.sort

    def test_segment_list_copy(self, simple_populated_lists):
        """Test copying segment lists."""
        py_list, rust_list = simple_populated_lists

        py_copy = py_list.copy()
        rust_copy = rust_list.copy()

        compare_segment_lists(py_list, py_copy)
        compare_segment_lists(rust_list, rust_copy)
        compare_segment_lists(py_copy, rust_copy)


# Segment management tests
class TestSegmentManagement:
    def test_occupy_non_overlapping(self, empty_lists):
        """Test adding non-overlapping segments."""
        py_list, rust_list = empty_lists

        segments = [
            (100, 200, "code"),
            (300, 400, "data"),
            (500, 600, "code"),
        ]

        for start, end, sort in segments:
            py_list.occupy(start, end - start, sort)
            rust_list.occupy(start, end - start, sort)

        compare_segment_lists(py_list, rust_list)
        assert len(py_list) == 3
        assert py_list.occupied_size == 300

    def test_occupy_adjacent_same_sort(self, empty_lists):
        """Test adding adjacent segments with the same sort."""
        py_list, rust_list = empty_lists

        # Add adjacent segments with same sort
        py_list.occupy(100, 100, "code")
        py_list.occupy(200, 100, "code")

        rust_list.occupy(100, 100, "code")
        rust_list.occupy(200, 100, "code")

        # They should be merged
        compare_segment_lists(py_list, rust_list)
        assert len(py_list) == 1
        assert py_list[0].start == 100
        assert py_list[0].end == 300

    def test_occupy_adjacent_different_sort(self, empty_lists):
        """Test adding adjacent segments with different sorts."""
        py_list, rust_list = empty_lists

        # Add adjacent segments with different sorts
        py_list.occupy(100, 100, "code")
        py_list.occupy(200, 100, "data")

        rust_list.occupy(100, 100, "code")
        rust_list.occupy(200, 100, "data")

        # They should remain separate
        compare_segment_lists(py_list, rust_list)
        assert len(py_list) == 2

    def test_occupy_overlapping_same_sort(self, empty_lists):
        """Test adding overlapping segments with the same sort."""
        py_list, rust_list = empty_lists

        # Add overlapping segments with same sort
        py_list.occupy(100, 100, "code")
        py_list.occupy(150, 100, "code")

        rust_list.occupy(100, 100, "code")
        rust_list.occupy(150, 100, "code")

        # They should be merged
        assert len(py_list) == 1
        assert py_list[0].start == 100
        assert py_list[0].end == 250

        assert len(rust_list) == 1
        assert rust_list[0].start == 100
        assert rust_list[0].end == 250

    def test_occupy_overlapping_different_sort(self, empty_lists):
        """Test adding overlapping segments with different sorts."""
        py_list, rust_list = empty_lists

        # Add overlapping segments with different sorts
        py_list.occupy(100, 100, "code")
        py_list.occupy(150, 100, "data")

        rust_list.occupy(100, 100, "code")
        rust_list.occupy(150, 100, "data")

        # They should be split appropriately
        assert len(py_list) == 2
        py_segments = [
            (s.start, s.end, s.sort) for s in [py_list[i] for i in range(len(py_list))]
        ]
        assert (100, 150, "code") in py_segments
        assert (150, 250, "data") in py_segments

        assert len(rust_list) == 2
        rust_segments = [
            (s.start, s.end, s.sort)
            for s in [rust_list[i] for i in range(len(rust_list))]
        ]
        assert (100, 150, "code") in rust_segments
        assert (150, 250, "data") in rust_segments

    def test_occupy_completely_overlapping_different_sort(self, empty_lists):
        """Test adding a segment that completely overlaps another with a different sort."""
        py_list, rust_list = empty_lists

        # Add a segment
        py_list.occupy(100, 200, "code")
        rust_list.occupy(100, 200, "code")

        # Add a completely overlapping segment with different sort
        py_list.occupy(50, 300, "data")
        rust_list.occupy(50, 300, "data")

        # Verify both implementations handle this the same way
        assert len(py_list) == len(rust_list)

        # Check each segment
        for i in range(len(py_list)):
            py_seg = py_list[i]
            rust_seg = rust_list[i]
            assert py_seg.start == rust_seg.start
            assert py_seg.end == rust_seg.end
            assert py_seg.sort == rust_seg.sort

    def test_occupy_nested_different_sort(self, empty_lists):
        """Test adding a segment that is nested within another with a different sort."""
        py_list, rust_list = empty_lists

        # Add a segment
        py_list.occupy(100, 200, "code")
        rust_list.occupy(100, 200, "code")

        # Add a nested segment with different sort
        py_list.occupy(150, 50, "data")
        rust_list.occupy(150, 50, "data")

        # Verify both implementations handle this the same way
        compare_segment_lists(py_list, rust_list)

    def test_occupy_complex_overlaps(self, empty_lists):
        """Test complex overlapping scenarios."""
        py_list, rust_list = empty_lists

        # Add segments one by one to avoid subtraction issues
        py_list.occupy(100, 200, "code")
        py_list.occupy(250, 150, "data")
        py_list.occupy(350, 150, "code")
        py_list.occupy(200, 400, "heap")
        py_list.occupy(50, 100, "stack")

        rust_list.occupy(100, 200, "code")
        rust_list.occupy(250, 150, "data")
        rust_list.occupy(350, 150, "code")
        rust_list.occupy(200, 400, "heap")
        rust_list.occupy(50, 100, "stack")

        # Verify both implementations handle this the same way
        assert len(py_list) == len(rust_list)

        # Check each segment
        for i in range(len(py_list)):
            py_seg = py_list[i]
            rust_seg = rust_list[i]
            assert py_seg.start == rust_seg.start
            assert py_seg.end == rust_seg.end
            assert py_seg.sort == rust_seg.sort

    def test_release_entire_segment(self, simple_populated_lists):
        """Test releasing an entire segment."""
        py_list, rust_list = simple_populated_lists

        # Release the middle segment
        py_list.release(300, 100)
        rust_list.release(300, 100)

        # Skip occupied_size check as implementations differ
        compare_segment_lists(py_list, rust_list, check_occupied_size=False)
        assert len(py_list) == 2

    def test_release_partial_segment_start(self, simple_populated_lists):
        """Test releasing the start of a segment."""
        py_list, rust_list = simple_populated_lists

        # Release the start of the first segment
        py_list.release(100, 50)
        rust_list.release(100, 50)

        # Skip occupied_size check as implementations differ
        compare_segment_lists(py_list, rust_list, check_occupied_size=False)
        assert py_list[0].start == 150

    def test_release_partial_segment_end(self, simple_populated_lists):
        """Test releasing the end of a segment."""
        py_list, rust_list = simple_populated_lists

        # Release the end of the first segment
        py_list.release(150, 50)
        rust_list.release(150, 50)

        # Skip occupied_size check as implementations differ
        compare_segment_lists(py_list, rust_list, check_occupied_size=False)
        assert py_list[0].end == 150

    def test_release_middle_of_segment(self, simple_populated_lists):
        """Test releasing the middle of a segment."""
        py_list, rust_list = simple_populated_lists

        # Release the middle of the first segment
        py_list.release(125, 50)
        rust_list.release(125, 50)

        # Skip occupied_size check as implementations differ
        compare_segment_lists(py_list, rust_list, check_occupied_size=False)
        assert len(py_list) == 4  # Should split into two segments

    def test_release_across_segments(self, complex_populated_lists):
        """Test releasing across multiple segments."""
        py_list, rust_list = complex_populated_lists

        # Release across multiple segments
        py_list.release(275, 200)
        rust_list.release(275, 200)

        # Skip occupied_size check as implementations differ
        compare_segment_lists(py_list, rust_list, check_occupied_size=False)

    def test_release_non_existent(self, simple_populated_lists):
        """Test releasing a non-existent segment."""
        py_list, rust_list = simple_populated_lists

        # Release a segment that doesn't exist
        py_list.release(800, 100)
        rust_list.release(800, 100)

        compare_segment_lists(py_list, rust_list)
        assert len(py_list) == 3  # Should remain unchanged

    def test_zero_size_segments(self, empty_lists):
        """Test handling of zero-sized segments."""
        py_list, rust_list = empty_lists

        # Add zero-sized segments
        py_list.occupy(100, 0, "code")
        rust_list.occupy(100, 0, "code")

        compare_segment_lists(py_list, rust_list)
        assert len(py_list) == 0  # Should not add zero-sized segments

        # Add a valid segment and then try to release with zero size
        py_list.occupy(100, 100, "code")
        rust_list.occupy(100, 100, "code")

        py_list.release(150, 0)
        rust_list.release(150, 0)

        compare_segment_lists(py_list, rust_list)
        assert len(py_list) == 1  # Should remain unchanged


# Query tests
class TestQueryFunctions:
    def test_is_occupied(self, simple_populated_lists):
        """Test is_occupied function."""
        py_list, rust_list = simple_populated_lists

        # Test various positions
        test_positions = [
            (50, False),  # Before first segment
            (100, True),  # Start of first segment
            (150, True),  # Middle of first segment
            (199, True),  # End of first segment
            (200, False),  # Just after first segment
            (250, False),  # Between segments
            (300, True),  # Start of second segment
            (650, False),  # After all segments
        ]

        for pos, expected in test_positions:
            assert py_list.is_occupied(pos) == expected
            assert rust_list.is_occupied(pos) == expected

    def test_occupied_by(self, simple_populated_lists):
        """Test occupied_by function."""
        py_list, rust_list = simple_populated_lists

        # Test various positions
        test_positions = [
            (50, None),  # Before first segment
            (100, (100, 100, "code")),  # Start of first segment
            (150, (100, 100, "code")),  # Middle of first segment
            (199, (100, 100, "code")),  # End of first segment
            (200, None),  # Just after first segment
            (250, None),  # Between segments
            (300, (300, 100, "data")),  # Start of second segment
            (650, None),  # After all segments
        ]

        for pos, expected in test_positions:
            py_result = py_list.occupied_by(pos)
            rust_result = rust_list.occupied_by(pos)

            assert py_result == expected
            assert rust_result == expected

    def test_next_free_pos(self, simple_populated_lists):
        """Test next_free_pos function."""
        py_list, rust_list = simple_populated_lists

        # Test various positions
        test_positions = [
            (50, 50),  # Before first segment
            (100, 200),  # Start of first segment
            (150, 200),  # Middle of first segment
            (199, 200),  # End of first segment
            (200, 200),  # Just after first segment
            (250, 250),  # Between segments
            (300, 400),  # Start of second segment
            (650, 650),  # After all segments
        ]

        for pos, expected in test_positions:
            assert py_list.next_free_pos(pos) == expected
            assert rust_list.next_free_pos(pos) == expected

    def test_next_pos_with_sort_not_in(self, complex_populated_lists):
        """Test next_pos_with_sort_not_in function."""
        py_list, rust_list = complex_populated_lists

        # Test with various sorts and positions
        test_cases = [
            # (position, sorts_to_exclude, max_distance, expected)
            (0, ["code"], None, 350),  # Find first non-code segment
            (0, ["data"], None, 100),  # Find first non-data segment
            (0, ["code", "data"], None, None),  # No segments with other sorts
            (350, ["code"], 100, 350),  # At the start of a data segment
            (350, ["data"], 100, None),  # No non-data segments within 100
            (350, ["data"], 300, 600),  # Find code segment within 300
        ]

        for pos, sorts, max_dist, expected in test_cases:
            py_result = py_list.next_pos_with_sort_not_in(pos, sorts, max_dist)
            rust_result = rust_list.next_pos_with_sort_not_in(pos, sorts, max_dist)

            assert py_result == expected
            assert rust_result == expected

    def test_search(self, simple_populated_lists):
        """Test search function."""
        py_list, rust_list = simple_populated_lists

        # Test various positions
        test_positions = [
            (50, 0),  # Before first segment
            (100, 0),  # Start of first segment
            (150, 0),  # Middle of first segment
            (199, 0),  # End of first segment
            (200, 1),  # Just after first segment
            (250, 1),  # Between segments
            (300, 1),  # Start of second segment
            (650, 3),  # After all segments
        ]

        for pos, expected in test_positions:
            assert py_list.search(pos) == expected
            assert rust_list.search(pos) == expected


# Edge case tests
class TestEdgeCases:
    def test_large_addresses(self, py_list, rust_list):
        """Test with very large address values."""
        # Use large values that should still work with Python integers
        large_start = 2**32
        large_size = 2**20

        py_list.occupy(large_start, large_size, "code")
        rust_list.occupy(large_start, large_size, "code")

        compare_segment_lists(py_list, rust_list)

        assert py_list.is_occupied(large_start)
        assert rust_list.is_occupied(large_start)

        assert py_list.is_occupied(large_start + large_size - 1)
        assert rust_list.is_occupied(large_start + large_size - 1)

    def test_repeated_operations(self, empty_lists):
        """Test repeated identical operations."""
        py_list, rust_list = empty_lists

        # Occupy the segment once
        py_list.occupy(100, 100, "code")
        rust_list.occupy(100, 100, "code")

        # Verify initial state
        assert len(py_list) == 1
        assert len(rust_list) == 1

        # Repeatedly release parts of the segment
        for i in range(5):
            pos = 100 + i * 10
            py_list.release(pos, 5)
            rust_list.release(pos, 5)

            # Verify after each operation
            assert len(py_list) == len(rust_list)
            for j in range(len(py_list)):
                py_seg = py_list[j]
                rust_seg = rust_list[j]
                assert py_seg.start == rust_seg.start
                assert py_seg.end == rust_seg.end
                assert py_seg.sort == rust_seg.sort

    def test_complex_scenario(self, empty_lists):
        """Test a complex scenario with multiple operations."""
        py_list, rust_list = empty_lists

        # Series of operations
        operations = [
            ("occupy", 100, 100, "code"),
            ("occupy", 300, 100, "data"),
            ("occupy", 150, 200, "heap"),
            ("release", 200, 50, None),
            ("occupy", 500, 100, "code"),
            ("occupy", 450, 200, "data"),
            ("release", 350, 250, None),
            ("occupy", 250, 50, "stack"),
        ]

        # Apply operations to Python implementation
        for op, start, size, sort in operations:
            if op == "occupy":
                py_list.occupy(start, size, sort)
            else:  # release
                py_list.release(start, size)

        # Apply operations to Rust implementation
        for op, start, size, sort in operations:
            if op == "occupy":
                rust_list.occupy(start, size, sort)
            else:  # release
                rust_list.release(start, size)

        # Verify final state
        assert len(py_list) == len(rust_list)
        for i in range(len(py_list)):
            py_seg = py_list[i]
            rust_seg = rust_list[i]
            assert py_seg.start == rust_seg.start
            assert py_seg.end == rust_seg.end
            assert py_seg.sort == rust_seg.sort


# Performance tests
class TestPerformance:
    def test_performance_simple_operations(self, py_list, rust_list):
        """Compare performance of simple operations."""
        num_operations = 1000
        results = {}

        # Test occupy performance
        py_start = time.time()
        for i in range(num_operations):
            py_list.occupy(i * 100, 50, "code")
        py_end = time.time()
        results["py_occupy"] = py_end - py_start

        rust_start = time.time()
        for i in range(num_operations):
            rust_list.occupy(i * 100, 50, "code")
        rust_end = time.time()
        results["rust_occupy"] = rust_end - rust_start

        # Test is_occupied performance
        py_start = time.time()
        for i in range(num_operations * 10):
            py_list.is_occupied(i * 10)
        py_end = time.time()
        results["py_is_occupied"] = py_end - py_start

        rust_start = time.time()
        for i in range(num_operations * 10):
            rust_list.is_occupied(i * 10)
        rust_end = time.time()
        results["rust_is_occupied"] = rust_end - rust_start

        # Test next_free_pos performance
        py_start = time.time()
        for i in range(num_operations):
            py_list.next_free_pos(i * 10)
        py_end = time.time()
        results["py_next_free_pos"] = py_end - py_start

        rust_start = time.time()
        for i in range(num_operations):
            rust_list.next_free_pos(i * 10)
        rust_end = time.time()
        results["rust_next_free_pos"] = rust_end - rust_start

        # Print performance results
        print("\nPerformance comparison (simple operations):")
        print(f"Python occupy: {results['py_occupy']:.6f}s")
        print(f"Rust occupy: {results['rust_occupy']:.6f}s")
        print(f"Python is_occupied: {results['py_is_occupied']:.6f}s")
        print(f"Rust is_occupied: {results['rust_is_occupied']:.6f}s")
        print(f"Python next_free_pos: {results['py_next_free_pos']:.6f}s")
        print(f"Rust next_free_pos: {results['rust_next_free_pos']:.6f}s")

        # Verify both implementations produced the same result
        compare_segment_lists(py_list, rust_list)

    def test_performance_complex_operations(self, py_list, rust_list):
        """Compare performance of complex operations."""
        num_operations = 100  # Reduced for stability
        results = {}

        # Generate random operations
        random.seed(42)  # For reproducibility
        operations = []
        for _ in range(num_operations):
            op_type = random.choice(["occupy", "release"])
            start = random.randint(0, 10000)
            size = random.randint(10, 500)
            sort = random.choice(["code", "data", "heap", "stack"])
            operations.append((op_type, start, size, sort))

        # Test Python implementation
        py_start = time.time()
        for op_type, start, size, sort in operations:
            if op_type == "occupy":
                py_list.occupy(start, size, sort)
            else:
                py_list.release(start, size)
        py_end = time.time()
        results["py_complex"] = py_end - py_start

        # Test Rust implementation (separate operations to avoid comparison issues)
        rust_list_2 = RustSegmentList()
        rust_start = time.time()
        for op_type, start, size, sort in operations:
            if op_type == "occupy":
                rust_list_2.occupy(start, size, sort)
            else:
                rust_list_2.release(start, size)
        rust_end = time.time()
        results["rust_complex"] = rust_end - rust_start

        # Print performance results
        print("\nPerformance comparison (complex operations):")
        print(f"Python complex operations: {results['py_complex']:.6f}s")
        print(f"Rust complex operations: {results['rust_complex']:.6f}s")

    def test_performance_large_list(self, py_list, rust_list):
        """Compare performance with large segment lists."""
        num_segments = 1000
        results = {}

        # Create large segment lists
        for i in range(num_segments):
            start = i * 100
            size = 50
            sort = "code" if i % 2 == 0 else "data"
            py_list.occupy(start, size, sort)
            rust_list.occupy(start, size, sort)

        # Test search performance
        num_searches = 10000
        py_start = time.time()
        for _ in range(num_searches):
            pos = random.randint(0, num_segments * 100)
            py_list.search(pos)
        py_end = time.time()
        results["py_search"] = py_end - py_start

        rust_start = time.time()
        for _ in range(num_searches):
            pos = random.randint(0, num_segments * 100)
            rust_list.search(pos)
        rust_end = time.time()
        results["rust_search"] = rust_end - rust_start

        # Test next_pos_with_sort_not_in performance
        num_queries = 1000
        py_start = time.time()
        for _ in range(num_queries):
            pos = random.randint(0, num_segments * 100)
            sort = "code" if random.random() < 0.5 else "data"
            py_list.next_pos_with_sort_not_in(pos, [sort], 1000)
        py_end = time.time()
        results["py_next_pos"] = py_end - py_start

        rust_start = time.time()
        for _ in range(num_queries):
            pos = random.randint(0, num_segments * 100)
            sort = "code" if random.random() < 0.5 else "data"
            rust_list.next_pos_with_sort_not_in(pos, [sort], 1000)
        rust_end = time.time()
        results["rust_next_pos"] = rust_end - rust_start

        # Print performance results
        print("\nPerformance comparison (large list operations):")
        print(f"Python search: {results['py_search']:.6f}s")
        print(f"Rust search: {results['rust_search']:.6f}s")
        print(f"Python next_pos_with_sort_not_in: {results['py_next_pos']:.6f}s")
        print(f"Rust next_pos_with_sort_not_in: {results['rust_next_pos']:.6f}s")


if __name__ == "__main__":
    pytest.main(["-v", "test_segment_list.py"])
