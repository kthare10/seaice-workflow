# Pegasus/HTCondor Debugging on FABRIC

Quick reference for monitoring and debugging Pegasus workflows running on HTCondor across FABRIC nodes.

## Connecting to Nodes

```bash
# SSH to submit node (replace IPv6 address with your submit node's)
ssh -i ~/.ssh/id_rsa -F ~/work/fabric_config/ssh_config ubuntu@<submit-ipv6>

# From submit node, SSH to a worker (uses /etc/hosts hostnames)
ssh -o StrictHostKeyChecking=no <worker-hostname>
```

## Workflow Monitoring

### Pegasus Status

```bash
# Overall workflow progress — running, queued, done, failed
pegasus-status

# Status of a specific run directory
pegasus-status <run-dir>

# Detailed analysis after completion or failure
pegasus-analyzer <run-dir>

# Full workflow statistics
pegasus-statistics <run-dir>
```

### Job State Log

The `jobstate.log` records every state transition with timestamps:

```bash
cat <run-dir>/jobstate.log

# Filter for a specific job
grep download_atl03 <run-dir>/jobstate.log

# Show only failures
grep JOB_FAILURE <run-dir>/jobstate.log
```

**Timestamps** are Unix epoch — convert with: `date -d @<timestamp>`

## HTCondor Queue

```bash
# List all jobs with status (I=idle, R=running, H=held, X=removed)
condor_q

# Show which worker each running job landed on
condor_q -run

# Verbose details for a specific job
condor_q -l <job_id>

# Why isn't a job matching any worker? (slot requirements analysis)
condor_q -better-analyze <job_id>

# Show only held jobs and their hold reasons
condor_q -held
```

## Job Output and Logs

### Live Output from Running Jobs

```bash
# Tail stdout of a running job (job ID from condor_q, e.g., 10.0)
condor_tail <job_id>

# Tail stderr
condor_tail <job_id> -stderr

# Follow stdout continuously
condor_tail -f <job_id>
```

### Completed Job Output

Job output files are in `<run-dir>/00/00/`:

```bash
# List all output/error files
ls <run-dir>/00/00/*.out.* <run-dir>/00/00/*.err.*

# View a specific job's output
cat <run-dir>/00/00/download_atl03_download_atl03.out.000

# View a specific job's error log
cat <run-dir>/00/00/download_atl03_download_atl03.err.000

# Check staging job logs (container/file transfers)
cat <run-dir>/00/00/stage_in_local_local_0_0.out.000
```

### HTCondor Job Logs

```bash
# Per-job HTCondor log (events, resource usage, errors)
cat <run-dir>/00/00/<job_name>.log

# Search for errors across all job logs
grep -r "Error\|FAILED\|Abnormal" <run-dir>/00/00/*.log
```

## Monitoring Download Progress

Download jobs (`download_atl03`, `download_sentinel2`) run on worker nodes and can take a long time. Here's how to track them:

### Check Which Worker Is Running the Download

```bash
# See which worker each job landed on (look for the HOST column)
condor_q -run

# Example output:
#  ID     OWNER   SUBMITTED    RUN_TIME  HOST(S)
#  10.0   ubuntu  3/19 20:32   0+00:33   slot1_1@GPN-gpu-worker-1
```

### Tail Download Output in Real Time

```bash
# Stream stdout from the running download job (job ID from condor_q)
condor_tail <job_id>

# Stream stderr (shows logging/progress messages)
condor_tail <job_id> -stderr

# Follow continuously
condor_tail -f <job_id>
```

### Check File Sizes on the Worker

SSH from the submit node to the worker running the download and inspect the condor execute directory:

```bash
# List ATL03 granule files being downloaded (watch file sizes grow)
ssh -o StrictHostKeyChecking=no <worker-hostname> \
    'ls -lh /var/lib/condor/execute/dir_*/*.h5 2>/dev/null'

# Example output (528M done, 268M still growing):
#  -rw-r--r-- 1 ubuntu ubuntu 528M Mar 19 20:56 .../ATL03_20191101084913_05410512_007_01.h5
#  -rw-r--r-- 1 ubuntu ubuntu 268M Mar 19 21:06 .../ATL03_20191101193551_05480510_007_01.h5

# List Sentinel-2 scenes being downloaded
ssh -o StrictHostKeyChecking=no <worker-hostname> \
    'ls -lh /var/lib/condor/execute/dir_*/sentinel2_* 2>/dev/null'

# Watch file sizes update every 10 seconds
ssh -o StrictHostKeyChecking=no <worker-hostname> \
    'watch -n 10 "ls -lh /var/lib/condor/execute/dir_*/*.h5 2>/dev/null"'
```

### Check Completed Download Output

After a download job finishes, its output is in the run directory:

```bash
# ATL03 download log (shows granule count, retry attempts, total photons)
cat <run-dir>/00/00/download_atl03_download_atl03.out.000

# Sentinel-2 download log (shows scene count, bands, cloud cover)
cat <run-dir>/00/00/download_sentinel2_download_sentinel2.out.000

# Check for download errors (SSL retries, HTTP failures)
cat <run-dir>/00/00/download_atl03_download_atl03.err.000
```

### Estimate Download Time

ATL03 granules are typically 500 MB–3 GB each. With `--max-granules 2`, expect 1–6 GB total. Sentinel-2 scenes are smaller (~50–200 MB per scene for 4 bands). Download speed depends on the worker's network path — workers with FabNetv4Ext (public IP) download faster than those routing through the management interface.

## HTCondor Pool Health

```bash
# Show all workers, slots, and their state (Claimed/Unclaimed/Owner)
condor_status

# Compact view
condor_status -compact

# Show only GPU-equipped slots
condor_status -compact -constraint 'TotalGPUs > 0'

# Show detailed attributes for a specific worker
condor_status -l <worker-hostname>

# Check collector daemon
condor_status -collector

# Verify all daemons are running
condor_who
```

## Checking Worker Nodes

### From the Submit Node

```bash
# Check files in the condor execute directory on a worker
ssh -o StrictHostKeyChecking=no <worker-hostname> \
    'ls -lh /var/lib/condor/execute/dir_*/ 2>/dev/null'

# Check download progress (e.g., ATL03 HDF5 files growing in size)
ssh -o StrictHostKeyChecking=no <worker-hostname> \
    'ls -lh /var/lib/condor/execute/dir_*/*.h5 2>/dev/null'

# Check disk usage on worker
ssh -o StrictHostKeyChecking=no <worker-hostname> \
    'df -h /var/lib/condor/execute/'

# Check worker memory/CPU usage
ssh -o StrictHostKeyChecking=no <worker-hostname> \
    'free -h && echo "---" && uptime'

# Check GPU status on GPU workers
ssh -o StrictHostKeyChecking=no <worker-hostname> \
    'nvidia-smi'
```

### Network Connectivity

```bash
# Ping a worker from submit (uses FABNet IPs via /etc/hosts)
ping -c 3 <worker-hostname>

# Test SSH connectivity
ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no <worker-hostname> hostname

# Check FABNet interface and assigned IP
ip addr show | grep 'inet ' | grep -v 127.0.0

# Verify /etc/hosts has correct entries
cat /etc/hosts | grep -v '^#' | grep -v '^$'

# Test port 22 reachability
nc -zv <worker-ip> 22 -w 5
```

## Common Issues

### Staging Jobs Taking Too Long

`stage_in_local_local_*` jobs transfer files (including container images) to workers:

```bash
# Check which staging jobs are slow
grep stage_in <run-dir>/jobstate.log

# View staging job output for transfer details
cat <run-dir>/00/00/stage_in_local_local_0_0.out.000
```

**Cause**: Usually the first run on a worker pulls the Singularity container from Docker Hub (several GB for GPU images). Pre-pull to avoid this:

```bash
# Run on each worker node
singularity pull --force /tmp/seaice.sif docker://kthare10/seaice-icesat2:latest
```

### Jobs Stuck in Idle

```bash
# Check why jobs aren't matching
condor_q -better-analyze <job_id>

# Common reasons:
# - No slots with enough memory (RequestMemory)
# - No slots with GPUs (RequestGPUs)
# - Workers not reporting to collector
condor_status  # verify workers are visible
```

### SSH Permission Denied Between Nodes

```bash
# Check if keys are exchanged
ssh -v -o ConnectTimeout=10 <worker-hostname> hostname 2>&1 | grep -i 'offering\|auth'

# Verify the key fingerprints match
ssh-keygen -lf /home/ubuntu/.ssh/id_rsa.pub                          # on source
grep '<source-hostname>' /home/ubuntu/.ssh/authorized_keys | ssh-keygen -lf -  # on target

# Check for duplicate IPs (same-site nodes sharing a FABNet subnet)
ip addr show | grep 'inet ' | grep -v 127.0.0
```

### IP Address Conflicts (Same-Site Nodes)

When multiple nodes are at the same FABRIC site, they share a FABNet subnet. If netplan assigned duplicate IPs, nodes SSH to themselves instead of peers:

```bash
# Check for duplicate/secondary IPs
ip addr show <fabnet-interface> | grep 'inet '

# Should show only ONE IP per node. If you see "secondary", fix it:
sudo ip addr del <duplicate-ip>/24 dev <interface>

# Then fix the netplan config permanently
sudo cat /etc/netplan/91-lan-route-<interface>.yaml
```

### Job Failures

```bash
# Quick summary of what failed and why
pegasus-analyzer <run-dir>

# Check the specific job's stderr
cat <run-dir>/00/00/<failed_job>.err.000

# Check HTCondor's view of the failure
condor_history <job_id> -l | grep -i 'exit\|reason\|err'

# Re-submit a failed workflow (resumes from where it stopped)
pegasus-run <run-dir>
```

## Useful Run Directory Paths

```
<run-dir>/
├── jobstate.log              # All job state transitions
├── braindump.yml             # Workflow metadata
├── catalogs/                 # Site, transformation, replica catalogs
├── seaice-0.dag.dagman.out   # DAGMan log (orchestrator)
├── 00/00/                    # Job submit files and output
│   ├── *.sub                 # HTCondor submit files
│   ├── *.out.000             # Job stdout
│   ├── *.err.000             # Job stderr
│   └── *.log                 # HTCondor job event log
└── monitord.log              # Pegasus monitoring daemon log
```
