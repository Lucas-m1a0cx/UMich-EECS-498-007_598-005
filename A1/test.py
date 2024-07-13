def find_best_k(k_to_accuracies):
    """
    Find the key with the highest average value in a dictionary where
    each key maps to a list of values. If there are ties, return the smallest key.

    Args:
        k_to_accuracies: Dictionary mapping values of k to lists of accuracies.

    Returns:
        best_k: The key with the highest average value. If there are ties,
                return the smallest key.
    """
    best_k = None
    best_avg = float('-inf')

    for k, accuracies in k_to_accuracies.items():
        avg_accuracy = sum(accuracies) / len(accuracies)
        if avg_accuracy > best_avg or (avg_accuracy == best_avg and (best_k is None or k < best_k)):
            best_avg = avg_accuracy
            best_k = k

    return best_k

# 示例使用
k_to_accuracies = {
    1: [0.9, 0.85, 0.88],
    3: [0.92, 0.91, 0.93],
    5: [0.89, 0.90, 0.91],
    8: [0.92, 0.93, 0.91],  # 相同的最大平均值
    10: [0.86, 0.85, 0.87]
}

best_k = find_best_k(k_to_accuracies)
print("最佳的k值是:", best_k)
