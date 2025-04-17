#!/bin/bash
#SBATCH --job-name=rdma_dedup_sweep
#SBATCH --nodelist=ng20101,ng20102
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=dedup_sweep_%j.log

# Shared config
CHUNK_SIZE=$((1024 * 1024))
THRESHOLD=$((CHUNK_SIZE))
RANDOMNESS=1.0
SERVER_PORT=13337
PEER_PORT=9090

# Temporary file to store the server IP
SERVER_IP_FILE=/scratch/$USER/server_ip_$SLURM_JOB_ID.txt

# Identify nodes
SERVER_NODE=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
CLIENT_NODE=$(scontrol show hostname $SLURM_NODELIST | tail -n 1)

# Write server IP to file
srun --ntasks=1 --nodelist=$SERVER_NODE bash -c "hostname -i | awk '{print \$1}' > $SERVER_IP_FILE" &
wait
SERVER_IP=$(cat $SERVER_IP_FILE)

echo "Server Node: $SERVER_NODE"
echo "Client Node: $CLIENT_NODE"
echo "Server IP: $SERVER_IP"

# Start the server once
srun --ntasks=1 --nodelist=$SERVER_NODE ./ucx_rdma_server \
    -p $SERVER_PORT -t $((1024 * 1024 * 1024)) -d 1 &
SERVER_PID=$!

sleep 3  # Ensure server starts

# Sweep over tuple counts
for TUPLE_COUNT in $((1024 * 1024)) $((2 * 1024 * 1024)) $((4 * 1024 * 1024)) $((8 * 1024 * 1024)); do
  echo "=== Testing with TUPLE_COUNT = $TUPLE_COUNT ==="

  # Launch Client 1
  srun --ntasks=1 --nodelist=$CLIENT_NODE ./deduplicate \
      -t $TUPLE_COUNT -c $CHUNK_SIZE -s $SERVER_IP -p $SERVER_PORT \
      -1 0 -2 1 -b $THRESHOLD -r $RANDOMNESS &
  CLIENT1_PID=$!

  sleep 2  # Allow initialization

  # Launch Client 2
  srun --ntasks=1 --nodelist=$CLIENT_NODE ./deduplicate \
      -t $TUPLE_COUNT -c $CHUNK_SIZE -s $SERVER_IP -p $SERVER_PORT \
      -1 0 -2 1 -b $THRESHOLD -S $SERVER_IP -P $PEER_PORT -r $RANDOMNESS &
  CLIENT2_PID=$!

  wait $CLIENT1_PID
  wait $CLIENT2_PID

  echo "--- Finished test for $TUPLE_COUNT tuples ---"
  sleep 5
done

# Wait for server process before cleaning up
kill $SERVER_PID
wait $SERVER_PID
rm -f $SERVER_IP_FILE
