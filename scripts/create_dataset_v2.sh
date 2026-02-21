#!/bin/bash
set -e

# ==================================================================================
# Create Dataset 2: processed_data/traffic_v2
# Traffic Types: 0(Chat), 2(FileTransfer), 4(VoIP), 5(VPN:Chat), 6(VPN:FileTransfer), 8(VPN:Streaming)
# ==================================================================================

SOURCE_DIR="processed_data"
TARGET_DIR="processed_data/traffic_v2"

echo "==================================================================="
echo "--- Creating Dataset 2: traffic_v2 ---"
echo "==================================================================="
echo "Traffic types: 0, 2, 4, 5, 6, 8"
echo ""

mkdir -p "$TARGET_DIR"

# Step 1: Copy QQ and Weixin data (Types 0, 2, 4)
echo "Step 1: Copying QQ and Weixin data..."
for app in qq weixin; do
    echo "  Copying $app..."
    cp -R "$SOURCE_DIR/traffic/$app" "$TARGET_DIR/"
done

# Step 2: Select VPN files for Type 5 (VPN: Chat)
echo ""
echo "Step 2: Copying VPN Chat files (Type 5)..."
mkdir -p "$TARGET_DIR/vpn_chat"
for vpn_file in vpn_aim_chat1a vpn_skype_chat1a vpn_icq_chat1a; do
    echo "  Copying $vpn_file..."
    cp -R "$SOURCE_DIR/vpn/$vpn_file.pcap.transformed"* "$TARGET_DIR/vpn_chat/"
done

# Step 3: Select VPN files for Type 6 (VPN: File Transfer)
echo ""
echo "Step 3: Copying VPN File Transfer files (Type 6)..."
mkdir -p "$TARGET_DIR/vpn_filetransfer"
for vpn_file in vpn_sftp_A vpn_skype_files1a; do
    echo "  Copying $vpn_file..."
    cp -R "$SOURCE_DIR/vpn/$vpn_file.pcap.transformed"* "$TARGET_DIR/vpn_filetransfer/"
done

# Step 4: Select VPN files for Type 8 (VPN: Streaming)
echo ""
echo "Step 4: Copying VPN Streaming files (Type 8)..."
mkdir -p "$TARGET_DIR/vpn_streaming"
for vpn_file in vpn_spotify_A vpn_vimeo_A vpn_youtube_A; do
    echo "  Copying $vpn_file..."
    cp -R "$SOURCE_DIR/vpn/$vpn_file.pcap.transformed"* "$TARGET_DIR/vpn_streaming/"
done

echo ""
echo "==================================================================="
echo "--- Dataset 2 Created Successfully! ---"
echo "Location: $TARGET_DIR"
echo "==================================================================="
echo ""
echo "Summary:"
echo "  - qq, weixin (Types: 0, 2, 4)"
echo "  - vpn_chat (Type 5): vpn_aim_chat, vpn_skype_chat, vpn_icq_chat"
echo "  - vpn_filetransfer (Type 6): vpn_sftp, vpn_skype_files"
echo "  - vpn_streaming (Type 8): vpn_spotify, vpn_vimeo, vpn_youtube"
echo ""
