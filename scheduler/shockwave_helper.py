class MinMaxSumKSubarrays(object):
    def solve(self, array, k):
        min_maxsum_k_subarrays = self.min_ksubarrays_maxsum(array, k)
        k_subarrays = self.split_ksubarrays_min_maxsum(
            array, min_maxsum_k_subarrays, k
        )
        return k_subarrays

    def min_ksubarrays_maxsum_check_mid(self, mid, array, K):
        count = 0
        sum = 0
        n = len(array)

        for i in range(n):
            if array[i] > mid:
                return False
            sum += array[i]
            if sum > mid:
                count += 1
                sum = array[i]
        count += 1
        if count <= K:
            return True
        return False

    def min_ksubarrays_maxsum(self, array, k):
        n = len(array)
        assert n > 0
        start = max(array)
        end = 0
        for i in range(n):
            end += array[i]
        min_maxsum = 0
        while start <= end:
            mid = (start + end) // 2
            if self.min_ksubarrays_maxsum_check_mid(mid, array, k):
                min_maxsum = mid
                end = mid - 1
            else:
                start = mid + 1
        return min_maxsum

    def split_ksubarrays_min_maxsum(self, array, min_maxsum, k):
        if k >= len(array):
            return [[elem] for elem in array]
        count = 0
        subarray_sum = 0
        n = len(array)
        k_splits = []
        splitl = 0
        for splitr in range(n):
            subarray_sum += array[splitr]
            if subarray_sum > min_maxsum:
                count += 1
                k_splits.append(array[splitl:splitr])
                subarray_sum = array[splitr]
                splitl = splitr
        count += 1
        k_splits.append(array[splitl:])

        while len(k_splits) < k:
            max_cardinality = max([len(split) for split in k_splits])
            assert max_cardinality > 1
            for idx in range(len(k_splits)):
                split = k_splits[idx]
                if sum(split) <= min_maxsum and len(split) == max_cardinality:
                    break
            nelems = len(split)
            left = split[0 : nelems // 2]
            right = split[nelems // 2 : nelems]
            k_splits = k_splits[0:idx] + [left, right] + k_splits[idx + 1 :]
            k_splits = [split for split in k_splits if len(split) > 0]

        return k_splits
