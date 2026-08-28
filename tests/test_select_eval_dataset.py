import csv
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from scripts.select_eval_dataset import (
    ImageRecord,
    select_records,
    write_metadata_subsets,
)


def record(
    split: str,
    group: str,
    item_id: str,
    *,
    width: int = 1200,
    height: int = 1200,
) -> ImageRecord:
    return ImageRecord(
        split=split,
        group=group,
        product_type="PRODUCT",
        item_id=item_id,
        image_id=f"image-{item_id}",
        source_path=Path(f"/{split}/{group}/PRODUCT__image-{item_id}.jpg"),
        width=width,
        height=height,
    )


class SelectRecordsTest(unittest.TestCase):
    def test_selection_is_square_grouped_and_deterministic(self) -> None:
        records = []
        for group in ("가구", "조명"):
            for index in range(6):
                records.append(record("input", group, f"{group}-{index}"))
        # Non-square images are outside the fixed evaluation domain.
        records.append(record("input", "조명", "wide-input", width=1400, height=1000))

        first = select_records(records, total_per_split=8, seed="fixed-seed")
        second = select_records(records, total_per_split=8, seed="fixed-seed")

        self.assertEqual(first, second)
        self.assertEqual(Counter(r.group for r in first.input), Counter({"가구": 4, "조명": 4}))
        self.assertTrue(all(r.width == r.height for r in first.input))

    def test_duplicate_product_contributes_at_most_one_image(self) -> None:
        records = [
            record("input", "가구", "same-product"),
            ImageRecord(
                split="input",
                group="가구",
                product_type="PRODUCT",
                item_id="same-product",
                image_id="second-view",
                source_path=Path("/input/가구/second-view.jpg"),
                width=1200,
                height=1200,
            ),
            record("input", "가구", "input-only"),
        ]

        selected = select_records(records, total_per_split=2, seed="fixed-seed")

        self.assertEqual(len({r.item_id for r in selected.input}), 2)


class MetadataSubsetTest(unittest.TestCase):
    def test_writes_one_traceable_row_per_selected_image(self) -> None:
        selected_input = record("input", "가구", "input-item")
        selected = (selected_input,)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source"
            output = root / "output"
            source.mkdir()
            output.mkdir()

            base_fields = ["group", "product_type", "item_id", "image_id", "path", "width", "height"]
            base_rows = [
                {
                    "group": image.group,
                    "product_type": image.product_type,
                    "item_id": image.item_id,
                    "image_id": image.image_id,
                    "path": f"original/{image.image_id}.jpg",
                    "width": str(image.width),
                    "height": str(image.height),
                }
                for image in selected
            ]
            for filename in ("pool.csv", "index-main.csv"):
                with (source / filename).open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=base_fields)
                    writer.writeheader()
                    writer.writerows(base_rows)
                    if filename == "index-main.csv":
                        writer.writerow(base_rows[0])  # Duplicate source rows collapse to one selected row.

            feature_fields = ["path", "group", "ring_med", "ring_std", "ink", "sat"]
            with (source / "features.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=feature_fields)
                writer.writeheader()
                for image in selected:
                    writer.writerow(
                        {
                            "path": f"curated/{image.group}/{image.source_path.name}",
                            "group": image.group,
                            "ring_med": "255.0",
                            "ring_std": "0.0",
                            "ink": "0.0",
                            "sat": "0.0",
                        }
                    )

            summary = write_metadata_subsets(selected, source_root=source, output_root=output)

            self.assertEqual(summary["subset_counts"], {"features.csv": 1, "index-main.csv": 1, "pool.csv": 1})
            self.assertEqual(set(summary["source_csv_sha256"]), {"features.csv", "index-main.csv", "pool.csv"})
            for filename in ("pool.csv", "index-main.csv", "features.csv"):
                with (output / filename).open(encoding="utf-8", newline="") as handle:
                    rows = list(csv.DictReader(handle))
                self.assertEqual(len(rows), 1)
                self.assertEqual({row["split"] for row in rows}, {"input"})
                self.assertTrue(all(row["selected_path"].startswith("input/") for row in rows))


if __name__ == "__main__":
    unittest.main()
