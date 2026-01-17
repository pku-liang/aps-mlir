def flat_to_tiled_offset(flat_offset, W):
    """直接从展平的row-major offset计算tiling后的offset"""
    y = flat_offset // W
    x = flat_offset % W
    block_y = (flat_offset / W) // 8
    block_x = (flat_offset % W) // 8
    blocks_per_row = W // 8
    block_id = block_y * blocks_per_row + block_x
    local_y = y % 8 
    local_x = x % 8
    local_offset = local_y * 8 + local_x
    tiled_offset = block_id * (8 * 8) + local_offset
    return tiled_offset, block_id, (y, x)

# 验证
W = 16
for flat_offset in range(W**2):
    tiled_offset, block_id, (y, x) = flat_to_tiled_offset(flat_offset, W)
    print(f"原始offset {flat_offset} (y={y},x={x}) -> tiled offset {tiled_offset} (块{block_id})")