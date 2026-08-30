#!/usr/bin/env python

import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


def _pc(name: str):
    return getattr(pc, name)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Synthesize a Qwen3-235B-style gating profile parquet by duplicating "
            "a smaller profile's layers using a simple offset strategy."
        )
    )
    p.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Input parquet path (e.g. gating_qwen3_sharegptv3_155.parquet)",
    )
    p.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output parquet path",
    )
    p.add_argument(
        "--dst-num-layers",
        type=int,
        default=94,
        help="Target number of layers (Qwen3-235B uses 94)",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output if it already exists",
    )
    p.add_argument(
        "--compression",
        default="snappy",
        help="Parquet compression codec (default: snappy)",
    )
    return p


def synthesize_offset_duplicate_layers(
    *,
    input_path: Path,
    output_path: Path,
    dst_num_layers: int,
    overwrite: bool,
    compression: str,
) -> None:
    if not input_path.exists():
        raise FileNotFoundError(str(input_path))

    if output_path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Refusing to overwrite existing output: {output_path} (pass --overwrite)"
            )
        output_path.unlink()

    pf = pq.ParquetFile(str(input_path))

    layer_min: int | None = None
    layer_max: int | None = None
    for rg in range(pf.num_row_groups):
        t = pf.read_row_group(rg, columns=["layer"])
        arr = t["layer"]
        mn = _pc("min")(arr).as_py()
        mx = _pc("max")(arr).as_py()
        layer_min = mn if layer_min is None else min(layer_min, mn)
        layer_max = mx if layer_max is None else max(layer_max, mx)

    if layer_min is None or layer_max is None:
        raise RuntimeError("Failed to infer layer span from input parquet")
    if layer_min != 0:
        raise ValueError(f"Expected layer min == 0, got {layer_min}")

    src_num_layers = int(layer_max) + 1
    if dst_num_layers < src_num_layers:
        raise ValueError(
            f"dst_num_layers ({dst_num_layers}) must be >= src_num_layers ({src_num_layers})"
        )

    extra = dst_num_layers - src_num_layers

    writer: pq.ParquetWriter | None = None
    file_schema: pa.Schema | None = None
    total_rows_in = 0
    total_rows_out = 0
    duplicated_rows = 0

    try:
        for rg in range(pf.num_row_groups):
            t = pf.read_row_group(rg)
            if writer is None:
                file_schema = t.schema
                writer = pq.ParquetWriter(
                    str(output_path),
                    file_schema,
                    compression=compression,
                    use_dictionary=True,
                )

            total_rows_in += t.num_rows
            writer.write_table(t)
            total_rows_out += t.num_rows

            if extra > 0:
                layers = t["layer"]
                mask = _pc("less")(layers, pa.scalar(extra, type=layers.type))
                dup = t.filter(mask)
                if dup.num_rows:
                    if file_schema is None:
                        raise RuntimeError("Internal error: missing output schema")

                    new_layers = _pc("add")(dup["layer"], pa.scalar(src_num_layers, type=layers.type))
                    dup_fixed = pa.Table.from_arrays(
                        [new_layers if n == "layer" else dup[n] for n in file_schema.names],
                        schema=file_schema,
                    )
                    writer.write_table(dup_fixed)
                    duplicated_rows += dup_fixed.num_rows
                    total_rows_out += dup_fixed.num_rows
    finally:
        if writer is not None:
            writer.close()

    print(
        "done",
        {
            "input": str(input_path),
            "output": str(output_path),
            "src_num_layers": src_num_layers,
            "dst_num_layers": dst_num_layers,
            "duplicated_rows": int(duplicated_rows),
            "rows_in": int(total_rows_in),
            "rows_out": int(total_rows_out),
        },
    )


def main() -> None:
    args = _build_arg_parser().parse_args()
    synthesize_offset_duplicate_layers(
        input_path=args.input,
        output_path=args.output,
        dst_num_layers=args.dst_num_layers,
        overwrite=args.overwrite,
        compression=args.compression,
    )


if __name__ == "__main__":
    main()
