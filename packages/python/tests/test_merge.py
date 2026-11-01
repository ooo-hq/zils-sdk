import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from zils_sdk import merge


def bf16(values):
    """bf16 bytes of float32 values that bf16 holds exactly (the low 16 bits must be zero)."""
    bits = np.asarray(values, dtype="<f4").view("<u4")
    assert not (bits & 0xFFFF).any(), "test values must be exact in bf16"
    return (bits >> 16).astype("<u2").tobytes()


def write_safetensors(path, tensors):
    """tensors: {name: (dtype, shape, raw bytes)}; an independent writer for fixtures."""
    header, offset, blobs = {}, 0, []
    for name, (dtype, shape, raw) in tensors.items():
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [offset, offset + len(raw)],
        }
        offset += len(raw)
        blobs.append(raw)
    text = json.dumps(header).encode()
    text += b" " * (-len(text) % 8)
    path.write_bytes(struct.pack("<Q", len(text)) + text + b"".join(blobs))


def read_safetensors(path):
    """{name: (dtype, shape, float64 values)} for BF16/F32 files."""
    data = path.read_bytes()
    (size,) = struct.unpack("<Q", data[:8])
    header = json.loads(data[8 : 8 + size])
    out = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        begin, end = info["data_offsets"]
        raw = data[8 + size + begin : 8 + size + end]
        if info["dtype"] == "BF16":
            values = (np.frombuffer(raw, "<u2").astype("<u4") << 16).view("<f4")
        else:
            values = np.frombuffer(raw, "<f4")
        out[name] = (info["dtype"], info["shape"], values.astype(np.float64))
    return out


def model(root, tensors, files=None):
    root.mkdir(parents=True)
    write_safetensors(root / "model.safetensors", tensors)
    (root / "config.json").write_text(json.dumps({"model_type": "qwen3_5_text"}))
    for name, text in (files or {}).items():
        (root / name).write_text(text)
    return root


class MergeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def pair(self, a_values, b_values, a_name="layers.0.w", b_name="layers.0.w"):
        a = model(self.root / "a", {a_name: ("BF16", [len(a_values)], bf16(a_values))})
        b = model(self.root / "b", {b_name: ("BF16", [len(b_values)], bf16(b_values))})
        return a, b

    def test_linear_is_the_weighted_mean_in_model_a_layout(self):
        a, b = self.pair([1.0, 2.0, -4.0], [3.0, 4.0, 0.0])
        summary = merge.merge([a, b], self.root / "out")
        tensors = read_safetensors(self.root / "out" / "model.safetensors")
        self.assertEqual(tensors["layers.0.w"][0], "BF16")
        np.testing.assert_array_equal(tensors["layers.0.w"][2], [2.0, 3.0, -2.0])
        self.assertEqual(summary["merged"], 1)

    def test_weight_is_the_share_of_the_second_model(self):
        a, b = self.pair([0.0, 4.0], [4.0, 0.0])
        merge.merge([a, b], self.root / "out", weights=[0.75, 0.25])
        values = read_safetensors(self.root / "out" / "model.safetensors")["layers.0.w"][2]
        np.testing.assert_array_equal(values, [1.0, 3.0])

    def test_bf16_rounds_half_to_even(self):
        # 1.0 and the next bf16 value 1.0078125 average to an exact tie, which rounds to the even 1.0.
        a, b = self.pair([1.0, 1.0078125], [1.0078125, 1.0234375])
        merge.merge([a, b], self.root / "out")
        values = read_safetensors(self.root / "out" / "model.safetensors")["layers.0.w"][2]
        np.testing.assert_array_equal(values, [1.0, 1.015625])

    def test_slerp_follows_the_arc_between_directions(self):
        a, b = self.pair([1.0, 0.0], [0.0, 1.0])
        merge.merge([a, b], self.root / "out", method="slerp")
        values = read_safetensors(self.root / "out" / "model.safetensors")["layers.0.w"][2]
        np.testing.assert_allclose(values, [0.70703125, 0.70703125])  # bf16 of sin(pi/4)

    def test_slerp_of_parallel_tensors_is_linear(self):
        a, b = self.pair([1.0, 2.0], [2.0, 4.0])
        merge.merge([a, b], self.root / "out", method="slerp")
        values = read_safetensors(self.root / "out" / "model.safetensors")["layers.0.w"][2]
        np.testing.assert_array_equal(values, [1.5, 3.0])

    def test_prefixed_names_line_up_and_model_a_only_tensors_are_kept(self):
        a = model(
            self.root / "a",
            {
                "model.language_model.layers.0.w": ("BF16", [2], bf16([1.0, 1.0])),
                "model.visual.blocks.0.w": ("BF16", [2], bf16([8.0, 8.0])),
            },
        )
        b = model(self.root / "b", {"layers.0.w": ("BF16", [2], bf16([3.0, 3.0]))})
        summary = merge.merge([a, b], self.root / "out")
        tensors = read_safetensors(self.root / "out" / "model.safetensors")
        self.assertEqual(
            set(tensors), {"model.language_model.layers.0.w", "model.visual.blocks.0.w"}
        )
        np.testing.assert_array_equal(tensors["model.language_model.layers.0.w"][2], [2.0, 2.0])
        np.testing.assert_array_equal(tensors["model.visual.blocks.0.w"][2], [8.0, 8.0])
        self.assertEqual((summary["merged"], summary["kept"]), (1, 1))

    def test_second_model_extra_heads_and_vision_are_ignored(self):
        # A Qwen base carries mtp.* and lm_head; a d3 chassis has neither; a text-only first model has no vision.
        a = model(self.root / "a", {"layers.0.w": ("BF16", [1], bf16([1.0]))})
        b = model(
            self.root / "b",
            {
                "model.language_model.layers.0.w": ("BF16", [1], bf16([3.0])),
                "mtp.layers.0.w": ("BF16", [1], bf16([9.0])),
                "lm_head.weight": ("BF16", [1], bf16([9.0])),
                "model.visual.blocks.0.w": ("BF16", [1], bf16([9.0])),
            },
        )
        summary = merge.merge([a, b], self.root / "out")
        tensors = read_safetensors(self.root / "out" / "model.safetensors")
        self.assertEqual(set(tensors), {"layers.0.w"})
        np.testing.assert_array_equal(tensors["layers.0.w"][2], [2.0])
        self.assertEqual((summary["merged"], summary["kept"], summary["ignored"]), (1, 0, 3))
        recipe = json.loads((self.root / "out" / "merge.json").read_text())
        self.assertEqual(recipe["ignored"], 3)

    def test_float32_inputs_are_written_in_model_a_dtype(self):
        a = model(self.root / "a", {"w": ("BF16", [2], bf16([1.0, 2.0]))})
        b = model(self.root / "b", {"w": ("F32", [2], np.asarray([3.0, 4.0], "<f4").tobytes())})
        merge.merge([a, b], self.root / "out")
        dtype, _, values = read_safetensors(self.root / "out" / "model.safetensors")["w"]
        self.assertEqual(dtype, "BF16")
        np.testing.assert_array_equal(values, [2.0, 3.0])

    def test_sharded_models_keep_model_a_shards_and_index(self):
        a = self.root / "a"
        a.mkdir()
        write_safetensors(a / "s1.safetensors", {"x": ("BF16", [1], bf16([1.0]))})
        write_safetensors(a / "s2.safetensors", {"y": ("BF16", [1], bf16([2.0]))})
        index = {"metadata": {}, "weight_map": {"x": "s1.safetensors", "y": "s2.safetensors"}}
        (a / "model.safetensors.index.json").write_text(json.dumps(index))
        (a / "config.json").write_text("{}")
        b = model(
            self.root / "b", {"x": ("BF16", [1], bf16([3.0])), "y": ("BF16", [1], bf16([4.0]))}
        )
        merge.merge([a, b], self.root / "out")
        out = self.root / "out"
        self.assertEqual(json.loads((out / "model.safetensors.index.json").read_text()), index)
        np.testing.assert_array_equal(read_safetensors(out / "s1.safetensors")["x"][2], [2.0])
        np.testing.assert_array_equal(read_safetensors(out / "s2.safetensors")["y"][2], [3.0])

    def test_other_files_come_from_model_a_and_the_recipe_is_recorded(self):
        a = model(
            self.root / "a",
            {"w": ("BF16", [1], bf16([1.0]))},
            {"tokenizer.json": "a-tokenizer", "LICENSE": "a-licence"},
        )
        write_safetensors(a / "readout.safetensors", {"readout": ("BF16", [1], bf16([5.0]))})
        b = model(self.root / "b", {"w": ("BF16", [1], bf16([3.0]))}, {"LICENSE": "b-licence"})
        merge.merge([a, b], self.root / "out", method="slerp", weights=[0.75, 0.25])
        out = self.root / "out"
        self.assertEqual((out / "tokenizer.json").read_text(), "a-tokenizer")
        self.assertEqual((out / "LICENSE").read_text(), "a-licence")
        self.assertEqual((out / "LICENSE-model-2").read_text(), "b-licence")
        self.assertEqual(
            (out / "readout.safetensors").read_bytes(), (a / "readout.safetensors").read_bytes()
        )
        recipe = json.loads((out / "merge.json").read_text())
        self.assertEqual(recipe["method"], "slerp")
        self.assertEqual(recipe["weights"], [0.75, 0.25])
        self.assertEqual([m["source"] for m in recipe["models"]], [str(a), str(b)])

    def test_refusals_leave_no_output(self):
        cases = {
            "differs in shape": (
                {"w": ("BF16", [2], bf16([1.0, 1.0]))},
                {"w": ("BF16", [3], bf16([1.0, 1.0, 1.0]))},
            ),
            "not in the first model": (
                {"w": ("BF16", [1], bf16([1.0]))},
                {"w": ("BF16", [1], bf16([2.0])), "extra": ("BF16", [1], bf16([2.0]))},
            ),
            "share no weights": (
                {"w": ("BF16", [1], bf16([1.0]))},
                {"v": ("BF16", [1], bf16([2.0]))},
            ),
            "identical": (
                {"w": ("BF16", [1], bf16([1.0]))},
                {"w": ("BF16", [1], bf16([1.0]))},
            ),
            "dtype": (
                {"w": ("BF16", [1], bf16([1.0]))},
                {"w": ("I64", [1], struct.pack("<q", 1))},
            ),
        }
        for index, (message, (a_tensors, b_tensors)) in enumerate(cases.items()):
            with self.subTest(message):
                a = model(self.root / f"a{index}", a_tensors)
                b = model(self.root / f"b{index}", b_tensors)
                out = self.root / f"out{index}"
                with self.assertRaisesRegex(ValueError, message):
                    merge.merge([a, b], out)
                self.assertFalse(out.exists())

    def test_the_same_folder_twice_is_refused_before_anything_is_written(self):
        a, _ = self.pair([1.0], [2.0])
        with patch.object(merge, "encode", side_effect=AssertionError("wrote output")):
            with self.assertRaisesRegex(ValueError, "the same model"):
                merge.merge([a, self.root / "a" / ".." / "a"], self.root / "out")
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), ["a", "b"])

    def three(self, *values):
        return [
            model(self.root / name, {"w": ("BF16", [len(v)], bf16(v))})
            for name, v in zip("abcdef", values, strict=False)
        ]

    def test_three_models_average_evenly_by_default(self):
        models = self.three([1.0, 0.0], [2.0, 3.0], [6.0, 6.0])
        summary = merge.merge(models, self.root / "out")
        values = read_safetensors(self.root / "out" / "model.safetensors")["w"][2]
        np.testing.assert_array_equal(values, [3.0, 3.0])
        self.assertEqual(summary["merged"], 1)
        recipe = json.loads((self.root / "out" / "merge.json").read_text())
        self.assertEqual(len(recipe["models"]), 3)
        self.assertEqual(recipe["weights"], [1 / 3, 1 / 3, 1 / 3])

    def test_weights_are_portions_normalised_to_one(self):
        models = self.three([0.0], [4.0], [8.0])
        merge.merge(models, self.root / "out", weights=[1, 1, 2])
        np.testing.assert_array_equal(
            read_safetensors(self.root / "out" / "model.safetensors")["w"][2], [5.0]
        )

    def test_a_weight_some_models_lack_is_kept_from_the_first(self):
        a = model(
            self.root / "a",
            {"w": ("BF16", [1], bf16([1.0])), "visual.x": ("BF16", [1], bf16([8.0]))},
        )
        b = model(self.root / "b", {"w": ("BF16", [1], bf16([3.0]))})
        c = model(
            self.root / "c",
            {"w": ("BF16", [1], bf16([5.0])), "visual.x": ("BF16", [1], bf16([2.0]))},
        )
        summary = merge.merge([a, b, c], self.root / "out")
        tensors = read_safetensors(self.root / "out" / "model.safetensors")
        np.testing.assert_array_equal(tensors["w"][2], [3.0])
        np.testing.assert_array_equal(tensors["visual.x"][2], [8.0])
        self.assertEqual((summary["merged"], summary["kept"]), (1, 1))

    def test_model_counts_slerp_and_repeats_are_checked_before_writing(self):
        models = self.three([1.0], [2.0], [3.0], [4.0], [5.0], [6.0])
        cases = [
            (models[:1], {}, "2 to 5 models"),
            (models, {}, "2 to 5 models"),
            (models[:3], {"method": "slerp"}, "SLERP merges exactly two"),
            ([models[0], models[1], models[0]], {}, "same model"),
        ]
        for chosen, options, message in cases:
            with self.subTest(message), self.assertRaisesRegex(ValueError, message):
                merge.merge(chosen, self.root / "out", **options)
            self.assertFalse((self.root / "out").exists())

    def test_every_other_models_licence_is_copied(self):
        a = model(self.root / "a", {"w": ("BF16", [1], bf16([1.0]))}, {"LICENSE": "a"})
        b = model(self.root / "b", {"w": ("BF16", [1], bf16([2.0]))}, {"LICENSE": "b"})
        c = model(self.root / "c", {"w": ("BF16", [1], bf16([3.0]))}, {"LICENSE.md": "c"})
        merge.merge([a, b, c], self.root / "out")
        out = self.root / "out"
        self.assertEqual(
            [(out / n).read_text() for n in ("LICENSE", "LICENSE-model-2", "LICENSE-model-3.md")],
            ["a", "b", "c"],
        )

    def test_duplicate_names_after_alignment_are_refused(self):
        a = model(
            self.root / "a",
            {"model.w": ("BF16", [1], bf16([1.0])), "w": ("BF16", [1], bf16([1.0]))},
        )
        b = model(self.root / "b", {"w": ("BF16", [1], bf16([2.0]))})
        with self.assertRaisesRegex(ValueError, "same name"):
            merge.merge([a, b], self.root / "out")

    def test_an_existing_output_is_never_overwritten(self):
        a, b = self.pair([1.0], [2.0])
        (self.root / "out").mkdir()
        with self.assertRaises(FileExistsError):
            merge.merge([a, b], self.root / "out")

    def test_weights_and_method_are_checked(self):
        a, b = self.pair([1.0], [2.0])
        for weights in ([0.0, 1.0], [-0.5, 1.5], [float("nan"), 1.0], [1.0], [1.0, 1.0, 1.0]):
            with self.subTest(weights=weights), self.assertRaisesRegex(ValueError, "weights"):
                merge.merge([a, b], self.root / "out", weights=weights)
        with self.assertRaisesRegex(ValueError, "method"):
            merge.merge([a, b], self.root / "out", method="ties")

    def test_a_folder_resolves_to_itself_and_other_text_is_refused(self):
        a = model(self.root / "a", {"w": ("BF16", [1], bf16([1.0]))})
        self.assertEqual(merge.resolve(str(a)), a)
        with self.assertRaisesRegex(ValueError, "folder or a Hugging Face"):
            merge.resolve("not a repo id")


class MergeCliTests(unittest.TestCase):
    def test_zils_merge_writes_the_model_and_reports_it(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            a = model(root / "a", {"w": ("BF16", [2], bf16([1.0, 2.0]))})
            b = model(root / "b", {"w": ("BF16", [2], bf16([3.0, 4.0]))})
            env = {k: v for k, v in os.environ.items() if not k.startswith("ZILS_")}
            result = subprocess.run(
                [sys.executable, "-m", "zils_sdk.cli", "merge", str(a), str(b), "--out", str(root / "out"),
                 "--method", "slerp", "--weight", "0.5"],
                capture_output=True, text=True, env=env, timeout=60,
            )  # fmt: skip
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Merged 1 tensors", result.stdout)
            self.assertTrue((root / "out" / "model.safetensors").is_file())
            refused = subprocess.run(
                [sys.executable, "-m", "zils_sdk.cli", "merge", str(a), str(b), "--out", str(root / "x"),
                 "--weight", "1"],
                capture_output=True, text=True, env=env, timeout=60,
            )  # fmt: skip
            self.assertEqual(refused.returncode, 2)
            self.assertIn("between 0 and 1", refused.stderr)
            self.assertFalse((root / "x").exists())
            c = model(root / "c", {"w": ("BF16", [2], bf16([5.0, 6.0]))})
            three = subprocess.run(
                [sys.executable, "-m", "zils_sdk.cli", "merge", str(a), str(b), str(c), "--out", str(root / "soup"),
                 "--weights", "1,1,2"],
                capture_output=True, text=True, env=env, timeout=60,
            )  # fmt: skip
            self.assertEqual(three.returncode, 0, three.stderr)
            self.assertIn("Merged 1 tensors from 3 models", three.stdout)
            for extra, message in (
                (["--weight", "0.3"], "--weight is for two models"),
                (["--weights", "1,2"], "3 weights"),
            ):
                bad = subprocess.run(
                    [sys.executable, "-m", "zils_sdk.cli", "merge", str(a), str(b), str(c), "--out", str(root / "y"), *extra],
                    capture_output=True, text=True, env=env, timeout=60,
                )  # fmt: skip
                self.assertEqual(bad.returncode, 2, extra)
                self.assertIn(message, bad.stderr)


if __name__ == "__main__":
    unittest.main()
