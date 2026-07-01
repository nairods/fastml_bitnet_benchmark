import torch
import sys
import os

def inspect_state_dict(path, max_values=10):
    print("=" * 80)
    print(f"🔍 Inspecting state_dict file: {path}")
    print("=" * 80)

    # File size - auto-format based on magnitude
    file_size_bytes = os.path.getsize(path)
    if file_size_bytes < 1024:
        print(f"📁 File size: {file_size_bytes:,} bytes")
    elif file_size_bytes < 1024 * 1024:
        file_size_kb = file_size_bytes / 1024
        print(f"📁 File size: {file_size_bytes:,} bytes ({file_size_kb:.1f} KB)")
    else:
        file_size_mb = file_size_bytes / (1024 * 1024)
        print(f"📁 File size: {file_size_bytes:,} bytes ({file_size_mb:.2f} MB)")
    print()

    sd = torch.load(path, map_location="cpu")

    print(f"\n✅ Loaded state_dict with {len(sd)} entries\n")

    for key, value in sd.items():
        print("-" * 80)
        print(f"Key: {key}")

        # Tensor info
        if torch.is_tensor(value):
            print(f"  Shape: {tuple(value.shape)}")
            print(f"  Dtype: {value.dtype}")

            # Flatten and show first values
            flat = value.flatten()

            n = min(max_values, flat.numel())
            preview = flat[:n].tolist()

            print(f"  First {n} values:")
            print(f"   {preview}")

        else:
            print(f"  (Non-tensor value: {type(value)})")
            print(f"  Value: {value}")

    print("\n" + "=" * 80)
    print("✅ Done.")
    print("=" * 80)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\nUsage:")
        print("  python inspect_state_dict.py model.pt\n")
        sys.exit(1)

    path = sys.argv[1]
    inspect_state_dict(path)