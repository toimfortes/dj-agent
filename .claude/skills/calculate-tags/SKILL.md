---
name: calculate-tags
description: Write energy ratings to Rekordbox's My Tag system via direct DB write. Never writes to Comments.
---

# Calculate Tags

The only tag the agent writes is the **energy rating** (1-10). This goes to Rekordbox's **My Tag** system via direct DB write. Comments are never touched.

## How My Tags Work

My Tags are a DB-only feature — they cannot be set via XML import. The agent writes to the `DjmdMyTag` and `DjmdSongMyTag` tables in `master.db`.

**Requires Rekordbox to be closed.**

## Structure

The agent creates:
- **Energy** — parent category (under root, alongside Genre/Components/Situation)
- **Energy:1** through **Energy:10** — child tags (only the values needed)

Each track gets associated with its energy tag via `DjmdSongMyTag`.

## Implementation

```python
from pyrekordbox import Rekordbox6Database
from pyrekordbox.db6 import tables
import random, uuid
from datetime import datetime, timezone

def write_energy_mytags(db, results):
    now = datetime.now(timezone.utc)
    existing_tags = {t.Name: t for t in db.get_my_tag()}

    # Create Energy parent if needed
    if "Energy" not in existing_tags:
        root_tags = [t for t in db.get_my_tag() if t.ParentID == "root"]
        max_seq = max(t.Seq for t in root_tags) if root_tags else 0
        energy_parent = tables.DjmdMyTag(
            ID=str(random.randint(100000000, 4294967295)),
            Name="Energy", ParentID="root", Seq=max_seq + 1,
            Attribute=1, UUID=str(uuid.uuid4()),
            rb_data_status=0, rb_local_data_status=0,
            rb_local_deleted=0, rb_local_synced=0,
            rb_local_usn=100, created_at=now, updated_at=now,
        )
        db.session.add(energy_parent)
        db.session.flush()
        energy_parent_id = energy_parent.ID
    else:
        energy_parent_id = str(existing_tags["Energy"].ID)

    # Create Energy:N child tags as needed
    energy_tag_ids = {}
    for val in set(r["energy"] for r in results if r.get("energy")):
        tag_name = f"Energy:{val}"
        if tag_name not in existing_tags:
            child = tables.DjmdMyTag(
                ID=str(random.randint(100000000, 4294967295)),
                Name=tag_name, ParentID=energy_parent_id, Seq=val,
                Attribute=1, UUID=str(uuid.uuid4()),
                rb_data_status=0, rb_local_data_status=0,
                rb_local_deleted=0, rb_local_synced=0,
                rb_local_usn=100, created_at=now, updated_at=now,
            )
            db.session.add(child)
            energy_tag_ids[val] = child.ID
        else:
            energy_tag_ids[val] = str(existing_tags[tag_name].ID)

    db.session.flush()

    # Associate tracks
    for r in results:
        if not r.get("energy"):
            continue
        content_id = str(r["db_content_id"])
        my_tag_id = energy_tag_ids[r["energy"]]
        # Remove old energy tags, add new one
        for etid in energy_tag_ids.values():
            db.session.query(tables.DjmdSongMyTag).filter_by(
                ContentID=content_id, MyTagID=etid).delete()
        db.session.add(tables.DjmdSongMyTag(
            ID=str(random.randint(100000000, 4294967295)),
            MyTagID=my_tag_id, ContentID=content_id, TrackNo=1,
            UUID=str(uuid.uuid4()),
            rb_data_status=0, rb_local_data_status=0,
            rb_local_deleted=0, rb_local_synced=0,
            rb_local_usn=100, created_at=now, updated_at=now,
        ))

    db.session.commit()
```

## Custom Tag Rules

The user can also define custom rules during a session:
> "Tag everything over 135 BPM in A minor as 'late-night-techno'"

Custom tags are added as My Tags alongside the energy tag and saved to `memory.json` under `custom_tag_rules`.
