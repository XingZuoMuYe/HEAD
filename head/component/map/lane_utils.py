def count_adjacent_lanes(lane_id, road_network):
    """给定 lane_id 和 road_network，返回该车道的并行车道总数（含自身）"""
    if lane_id not in road_network.graph:
        return 1

    # 向左遍历找最左侧车道
    leftmost = lane_id
    left_visited = set()
    while True:
        if leftmost in left_visited:
            break
        left_visited.add(leftmost)

        lane_info = road_network.graph.get(leftmost)
        if not lane_info or not getattr(lane_info, "left_lanes", []):
            break

        for neighbor in lane_info.left_lanes:
            if neighbor in road_network.graph:
                leftmost = neighbor
                break
        else:
            break

    # 向右遍历计数
    count = 0
    current = leftmost
    right_visited = set()
    while True:
        if current in right_visited:
            break
        right_visited.add(current)
        count += 1

        lane_info = road_network.graph.get(current)
        if not lane_info or not getattr(lane_info, "right_lanes", []):
            break

        for neighbor in lane_info.right_lanes:
            if neighbor in road_network.graph:
                current = neighbor
                break
        else:
            break

    return count
