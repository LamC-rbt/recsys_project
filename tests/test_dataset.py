import pytest
from pathlib import Path


@pytest.fixture
def dataset_file():
    return Path("datasets/MovieLens_Large/ml-1m.txt")


def test_file_format(dataset_file):
    """Each line should be either empty or contain exactly two integers (user_id, item_id)."""
    assert dataset_file.exists(), f"Dataset file not found: {dataset_file}"

    user_ids = set()
    item_ids = set()

    with open(dataset_file, "r") as f:
        for line_no, line in enumerate(f, 1):
            parts = line.strip().split()
            if not parts:
                continue
            assert len(parts) == 2, f"Line {line_no} does not have 2 columns: {line.strip()}"
            user_id, item_id = map(int, parts)
            user_ids.add(user_id)
            item_ids.add(item_id)

    assert len(user_ids) > 0 and len(item_ids) > 0, "No valid user-item interactions found."

    max_user, max_item = max(user_ids), max(item_ids)
    for u in user_ids:
        assert 1 <= u <= max_user, f"Invalid user_id {u}"
    for i in item_ids:
        assert 1 <= i <= max_item, f"Invalid item_id {i}"
