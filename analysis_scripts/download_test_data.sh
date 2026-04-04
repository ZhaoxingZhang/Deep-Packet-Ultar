#!/bin/bash
# Download test data from remote server

SSH_HOST="featurize@workspace.featurize.cn"
SSH_PORT="54651"
SSH_PASS="9395a565"
REMOTE_DATA_DIR="/home/featurize/data/train_test_data"
LOCAL_DATA_DIR="./train_test_data"

echo "==================================================================="
echo "  Downloading Test Data from Remote Server"
echo "==================================================================="
echo ""
echo "Remote: $SSH_HOST:$SSH_PORT:$REMOTE_DATA_DIR"
echo "Local: $LOCAL_DATA_DIR"
echo ""

# Check if sshpass is installed
if ! command -v sshpass &> /dev/null; then
    echo "Error: sshpass is not installed."
    echo "Please install it first:"
    echo "  brew install sshpass  # macOS"
    echo "  sudo apt-get install sshpass  # Ubuntu/Debian"
    exit 1
fi

# Create local data directory
echo "Creating local data directory..."
mkdir -p $LOCAL_DATA_DIR

# Download test data
echo ""
echo "Downloading test data from remote server..."
echo "This may take a while depending on your connection speed."
echo ""

sshpass -p "$SSH_PASS" scp -P "$SSH_PORT" -r \
  "$SSH_HOST:$REMOTE_DATA_DIR/exp_traffic" \
  "$SSH_HOST:$REMOTE_DATA_DIR/exp_traffic_v2" \
  "$SSH_HOST:$REMOTE_DATA_DIR/exp_traffic_v3" \
  "$LOCAL_DATA_DIR/"

echo ""
echo "==================================================================="
echo "  Download Complete!"
echo "==================================================================="
echo ""
echo "Downloaded data:"
echo "  - $LOCAL_DATA_DIR/exp_traffic/"
echo "  - $LOCAL_DATA_DIR/exp_traffic_v2/"
echo "  - $LOCAL_DATA_DIR/exp_traffic_v3/"
echo ""
echo "You can now run the analysis scripts!"
