#!/bin/bash
#SBATCH --job-name=deduplication
#SBATCH --nodelist=cdr2631,cdr2632,cdr2637
#SBATCH --nodes=3
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
##SBATCH --gpus-per-node=2
#SBATCH --gres=gpu:v100l:2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=deduplication_%j.log

module load cuda/12.6
module load ucx/1.16

# Shared config
CHUNK_SIZE=$((1024 * 1024))
TUPLES_COUNTS=(
  $((1024 * 1024)) 
  $((2 * 1024 * 1024)) 
  $((4 * 1024 * 1024))
  $((8 * 1024 * 1024))
  $((16 * 1024 * 1024))
  $((32 * 1024 * 1024))
  $((64 * 1024 * 1024))
  $((128 * 1024 * 1024))
 #  $((256 * 1024 * 1024))
)

CHUNKS_COUNTS=(
  1
  2
  4
  8
  16
)

CHUNK_SIZES=(
  $((256 * 1024)) 
  $((2 * 256 * 1024)) 
  $((4 * 256 * 1024))
  $((8 * 256 * 1024))
  $((16 * 256 * 1024))
  $((32 * 256 * 1024))
  $((64 * 256 * 1024))
  $((128 * 256 * 1024))
  #$((256 * 256 * 1024))
)

# THRESHOLD=$((CHUNK_SIZE))
RANDOMNESS_VALUES=(
  0.01
  0.05
  0.1
  0.2
  0.4
  0.8
  1
)
SERVER_PORT=8000
PEER_PORT=9090

# Temporary file to store the server IP
SERVER_IP_FILE=/scratch/$USER/server_ip_$SLURM_JOB_ID.txt
CLIENT1_IP_FILE=/scratch/$USER/client1_ip_$SLURM_JOB_ID.txt
CLIENT2_IP_FILE=/scratch/$USER/client2_ip_$SLURM_JOB_ID.txt

# Identify nodes
SERVER_NODE=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
CLIENT1_NODE=$(scontrol show hostname $SLURM_NODELIST | tail -n 2 | head -n 1)
CLIENT2_NODE=$(scontrol show hostname $SLURM_NODELIST | tail -n 1)

# Write server IP to file
srun --ntasks=1 --nodelist=$SERVER_NODE bash -c "hostname -i | awk '{print \$1}' > $SERVER_IP_FILE" &
wait
SERVER_IP=$(cat $SERVER_IP_FILE)

# Write client1 IP to file
srun --ntasks=1 --nodelist=$CLIENT1_NODE bash -c "hostname -i | awk '{print \$1}' > $CLIENT1_IP_FILE" &
wait

# Write client2 IP to file
srun --ntasks=1 --nodelist=$CLIENT2_NODE bash -c "hostname -i | awk '{print \$1}' > $CLIENT2_IP_FILE" &
wait

CLIENT1_IP=$(cat $CLIENT1_IP_FILE)
CLIENT2_IP=$(cat $CLIENT2_IP_FILE)

echo "Server Node: $SERVER_NODE"
echo "Client1 Node: $CLIENT1_NODE"
echo "Client2 Node: $CLIENT2_NODE"
echo "Server IP: $SERVER_IP"
echo "Client1 IP: $CLIENT1_IP"
echo "Client2 IP: $CLIENT2_IP"

ROOT_DIR=$PWD/experiments
mkdir -p "$ROOT_DIR"

# Start the server once
#
for randomness in "${RANDOMNESS_VALUES[@]}"; do
  for (( i=0; i<${#TUPLES_COUNTS[@]}; i++ )); do 
    DIR="$ROOT_DIR/${TUPLES_COUNTS[i]}_tuples_${randomness}_randomness"
    mkdir -p "$DIR"

    echo "=== Testing with TUPLE_COUNT = ${TUPLES_COUNTS[i]}, RANDOMNESS = ${randomness} ==="
  
    srun --ntasks=1 --nodelist=$SERVER_NODE ./RDMA/ucx_rdma_server \
      -p $SERVER_PORT -t ${CHUNK_SIZES[i]} -1 ${CLIENT1_IP} -2 ${CLIENT2_IP}  &> "$DIR/server.log" &
  
    SERVER_PID=$!
  
    sleep 3  # Ensure server starts
  
    # Launch Client 1
    srun --ntasks=1 --nodelist=$CLIENT1_NODE ./deduplicate \
        -t ${TUPLES_COUNTS[i]} -c ${CHUNK_SIZES[i]} -s $SERVER_IP -p $SERVER_PORT \
        -1 0 -2 1 -b ${CHUNK_SIZES[i]} -r $randomness -e 9999 &> "$DIR/client1.log" &
    CLIENT1_PID=$!
  
    sleep 2  # Allow initialization
  
    # Launch Client 2
    srun --ntasks=1 --nodelist=$CLIENT2_NODE ./deduplicate \
        -t ${TUPLES_COUNTS[i]} -c ${CHUNK_SIZES[i]} -s $SERVER_IP -p $SERVER_PORT \
        -1 0 -2 1 -b ${CHUNK_SIZES[i]} -S $CLIENT1_IP -P $PEER_PORT -r $randomness -e 10000 &> "$DIR/client2.log" &
    CLIENT2_PID=$!
  
    wait $CLIENT1_PID
    wait $CLIENT2_PID
  
    wait $SERVER_PID
    echo "--- Finished test for ${TUPLES_COUNTS[i]} tuples ---"
    sleep 5
    ((SERVER_PORT++))
  done
done

rm -f $SERVER_IP_FILE
rm -f $CLIENT1_IP_FILE
