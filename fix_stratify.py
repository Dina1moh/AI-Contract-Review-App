with open("src/Run_evaluation.py", "r", encoding="utf-8") as f:
    content = f.read()
    
# Remove the broken block
broken_block = """            try:
                from sklearn.model_selection import train_test_split
                _, df = train_test_split(df, test_size=sample_size, stratify=df["clause_type"], random_state=42)
            except ValueError:
                # Fallback if classes are too small
                df = df.groupby('clause_type', group_keys=False).apply(
                    lambda x: x.sample(max(1, int(round(sample_size * len(x) / len(df)))), random_state=42)
                )
                df = df.sample(min(sample_size, len(df)), random_state=42)
"""
if broken_block in content:
    content = content.replace(broken_block, "")
    with open("src/Run_evaluation.py", "w", encoding="utf-8") as f:
        f.write(content)
        print("Fixed!")
else:
    print("Not found.")

