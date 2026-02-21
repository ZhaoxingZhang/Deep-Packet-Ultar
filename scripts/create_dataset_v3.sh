#!/bin/bash
set -e

# ==================================================================================
# Create Dataset 3: processed_data/traffic_v3
# Traffic Types: 0(Chat), 2(FileTransfer), 4(VoIP), 7(VPN:Email), 9(VPN:Torrent), 10(VPN:Voip)
# ==================================================================================

SOURCE_DIR="processed_data"
TARGET_DIR="processed_data/traffic_v3"

echo "==================================================================="
echo "--- Creating Dataset 3: traffic_v3 ---"
echo "==================================================================="
echo "Traffic types: 0, 2, 4, 7, 9, 10"
echo ""

mkdir -p "$TARGET_DIR"

# Step 1: Copy QQ and Weixin data (Types 0, 2, 4)
echo "Step 1: Copying QQ and Weixin data..."
for app in qq weixin; do
    echo "  Copying $app..."
    cp -R "$SOURCE_DIR/traffic/$app" "$TARGET_DIR/"
done

# Step 2: Select VPN files for Type 7 (VPN: Email)
echo ""
echo "Step 2: Copying VPN Email files (Type 7)..."
mkdir -p "$TARGET_DIR/vpn_email"
for vpn_file in vpn_email2a vpn_email2b; do
    echo "  Copying $vpn_file..."
    cp -R "$SOURCE_DIR/vpn/$vpn_file.pcap.transformed"* "$TARGET_DIR/vpn_email/"
done

# Step 3: Select VPN files for Type 9 (VPN: Torrent)
echo ""
echo "Step 3: Copying VPN Torrent files (Type 9)..."
mkdir -p "$TARGET_DIR/vpn_torrent"
echo "  Copying vpn_bittorrent..."
cp -R "$SOURCE_DIR/vpn/vpn_bittorrent.pcap.transformed"* "$TARGET_DIR/vpn_torrent/"

# Step 4: Select VPN files for Type 10 (VPN: Voip)
echo ""
echo "Step 4: Copying VPN VoIP files (Type 10)..."
mkdir -p "$TARGET_DIR/vpn_voip"
for vpn_file in vpn_skype_audio1 vpn_skype_audio2 vpn_facebook_audio2 vpn_voipbuster1a; do
    echo "  Copying $vpn_file..."
    cp -R "$SOURCE_DIR/vpn/$vpn_file.pcap.transformed"* "$TARGET_DIR/vpn_voip/"
done

echo ""
echo "==================================================================="
echo "--- Dataset 3 Created Successfully! ---"
echo "Location: $TARGET_DIR"
echo "==================================================================="
echo ""
echo "Summary:"
echo "  - qq, weixin (Types: 0, 2, 4)"
echo "  - vpn_email (Type 7): vpn_email"
echo "  - vpn_torrent (Type 9): vpn_bittorrent"
echo "  - vpn_voip (Type 10): vpn_skype_audio, vpn_facebook_audio, vpn_voipbuster"
echo ""
