# Comment Base64 Parts

Put copied base64 chunk files in this directory.

Recommended names:

```text
part001.txt
part002.txt
part003.txt
...
```

Each file should contain only the base64 body copied from the cloud
`.part*.b64.txt` file. The restore script sorts files by name and concatenates
them in order.

Restore:

```bash
python ../restore_base64_comment_parts.py --parts-dir . --output ../restored_demo.png --force
```

If you copied chunks inside markdown code fences, add `--allow-fenced`.
