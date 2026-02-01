# Clarius Exam Recording Utility

## Setup

- Read `documentation/clarius_setup.md` for setting up the Clarius connection
- (Optional) If using VR tracking, ensure SteamVR is running with a connected tracker
- Make sure to turn on position data for the scanning session or IMU data will be empty (Use the option button below the scanning window)

## Usage

Run the async recording utility via the `cusg-record-async` command:

```bash
source scripts/setup.sh && cusg-record-async \
  --ip 192.168.1.1 --port 5828 \
  --outputdir ./output \
  --interval 50 \
  --writers 1
```

On Windows, see `run.example.bat` or:

```batch
call scripts\setup.bat
cusg-record-async --ip 66.66.66.160 --port 5828 --outputdir .\output --interval 50 --writers 1 --preserve-date
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--ip` | `192.168.1.1` | Clarius scanner IP address |
| `--port` | `5828` | Clarius scanner port |
| `--outputdir` | (required) | Base directory for recording output |
| `--interval` | `50` | Frame capture interval in milliseconds |
| `--writers` | `1` | Number of parallel image writer processes |
| `--preserve-date` | `true` | Preserve Clarius timestamp |
| `--enable-tracking` | `true` | Enable VR tracker pose capture |
| `--disable-tracking` | `false` | Disable VR tracker pose capture |
| `--debug` | `false` | Enable debug output |

## Keyboard Controls

At startup, the utility prompts for operator and patient names, then waits for user input.

| Key | Action |
|---|---|
| `Enter` | Toggle recording start/stop |
| `Space` (hold) | Mark current frames as **valid** |
| Any letter/number (hold) | Set **reference** label to that character |
| `q` | Quit the application |

- **Valid** and **reference** labels are recorded per-frame in the metadata CSV
- Reference labels persist while the key is held; releasing clears the label
- Multiple recordings can be made in a single session (each `Enter` toggle creates a new recording directory)

## Output Structure

Each recording creates a timestamped directory under `--outputdir`:

```
outputdir/
  usgr_invivo_2025-01-15--14-30-45/
    images/
      frame_000000.png
      frame_000001.png
      ...
    frame.meta.csv
    recording.meta.json
```

- **`images/`** — Grayscale PNG frames written by worker processes
- **`frame.meta.csv`** — Per-frame metadata with columns: `index`, `timestamp`, IMU fields (`gx,gy,gz,ax,ay,az,mx,my,mz,qw,qx,qy,qz`), `clarius_timestamp`, `microns_per_pixel`, `valid`, `reference`, and pose fields (`r00,r01,r02,...,tz`)
- **`recording.meta.json`** — Recording-level metadata (operator name, patient name, tracking device serial, recording timestamp)

## Legacy Utility

The original synchronous recording utility is still available via `cusg-record`. See `record.py` for details.