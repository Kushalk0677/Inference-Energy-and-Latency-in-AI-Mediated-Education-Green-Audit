# Re-decode all responses from output_ids if still in memory
# If not, at minimum save what you have

df = pd.DataFrame(results)
df['Response'] = df['Response'].astype(str)
df.to_csv("responses_backup.csv", index=False, encoding='utf-8-sig')
files.download("responses_backup.csv")
print(f"Saved {len(df)} rows")
print(f"Responses present: {(df['Response'] != '').sum()}")