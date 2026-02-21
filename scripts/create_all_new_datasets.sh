#!/bin/bash
set -e

# ==================================================================================
# Create All New Datasets
# ==================================================================================

echo "==================================================================="
echo "--- Creating All New Datasets ---"
echo "==================================================================="
echo ""

# Create Dataset 2
echo ">>> Creating Dataset 2 (traffic_v2)..."
bash scripts/create_dataset_v2.sh

echo ""
echo "==================================================================="
echo ""

# Create Dataset 3
echo ">>> Creating Dataset 3 (traffic_v3)..."
bash scripts/create_dataset_v3.sh

echo ""
echo "==================================================================="
echo "--- All Datasets Created Successfully! ---"
echo "==================================================================="
echo ""
echo "Dataset Summary:"
echo ""
echo "Dataset 1 (existing): processed_data/traffic"
echo "  Types: 0, 2, 4, 6, 7, 8, 9, 10 (8 types)"
echo "  VPN: ftps, email, netflix, bittorrent, hangouts_audio"
echo ""
echo "Dataset 2 (new): processed_data/traffic_v2"
echo "  Types: 0, 2, 4, 5, 6, 8 (6 types)"
echo "  VPN: aim/skype/icq_chat, sftp/skype_files, spotify/vimeo/youtube"
echo ""
echo "Dataset 3 (new): processed_data/traffic_v3"
echo "  Types: 0, 2, 4, 7, 9, 10 (6 types)"
echo "  VPN: email, bittorrent, skype/facebook_audio, voipbuster"
echo ""
echo "==================================================================="
