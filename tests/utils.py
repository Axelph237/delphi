def print_tree(root, details: str = "", recurse_symbol="children", indent=0):
    prefix = "  " * indent

    print(f"{prefix}{type(root).__name__}: {root!r}")
    print(f"{prefix}{details}")

    for child in getattr(root, recurse_symbol, []):
        print_tree(child, details, recurse_symbol, indent + 1)