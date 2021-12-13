#1.有序序列
#2.找到middle,left和right
#3比较大小移动left和right
def binarySearch(nums:List[int], target:int, lower:bool) -> int:
    left, right = 0, len(nums)-1
    ans = len(nums)
    while left<=right: # 不变量：左闭右闭区间
        middle = left + (right-left) //2 
        # lower为True，执行前半部分，找到第一个大于等于 target的下标 ，否则找到第一个大于target的下标
        if nums[middle] > target or (lower and nums[middle] >= target): 
            right = middle - 1
            ans = middle
        else: 
            left = middle + 1
    return ans
