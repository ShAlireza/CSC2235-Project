#!/bin/bash
#SBATCH --job-name=pipeline_job  # Job name
#SBATCH --nodelist=ng20101,ng20102    # Nodes to run the job on
#SBATCH --nodes=2                     # Number of nodes
#SBATCH --ntasks-per-node=1           # Number of tasks per node
#SBATCH --gpus-per-node=1             # Number of GPUs per node
#SBATCH --mem=8G                     # Memory per node
#SBATCH --time=01:00:00               # Time limit hrs:min:sec
#SBATCH --output=job_output_%j.log    # Standard output and error log

# Load required modules
# module load cuda/12.6
# module load ucx/1.16

# Temporary file to store the server IP
SERVER_IP_FILE=/scratch/$USER/server_ip_$SLURM_JOB_ID.txt

# Get the server node (first node in the list)
SERVER_NODE=$(scontrol show hostname $SLURM_NODELIST | head -n 1)

# Get the client node (second node in the list)
CLIENT_NODE=$(scontrol show hostname $SLURM_NODELIST | tail -n 1)

# Run the server on the server node and write its IP to a shared file
srun --ntasks=1 --nodelist=$SERVER_NODE bash -c "hostname -i | awk '{print \$1}' > $SERVER_IP_FILE" &
SERVER_PID=$!


# Wait for the server's IP to be written
wait $SERVER_PID

# Read the server IP from the shared file
SERVER_IP=$(cat $SERVER_IP_FILE)


echo Server $SERVER_NODE
echo Client $CLIENT_NODE
echo SERVER IP $SERVER_IP

# Run the server on the server node
srun --ntasks=1 --nodelist=$SERVER_NODE ./server -s $((1)) -d $((1024 * 1024 * 1024)) &
SERVER_PID=$!

sleep 1
# Wait for both processes to complete
# wait $SERVER_PID

# Run the client on the client node with the server's IP address
srun --ntasks=1 --nodelist=$CLIENT_NODE ./client -s $((1)) -d $((1024 * 1024 * 1024)) -n $SERVER_IP &
CLIENT_PID=$!

# Wait for both processes to complete
wait $CLIENT_PID

# Cleanup the temporary file
rm -f $SERVER_IP_FILE

