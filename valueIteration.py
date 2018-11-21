S = [1, 2, 3, 4, 5, 6, 0, 7, 0, 8, 9, 10, 0, 0, 11, 0]
print(len(S))
A = ["n", "e", "s", "w"]
theta = 0.1
delta = 0
V = [0  for _ in range(16)]
print(V)

def dynamics(s, a): # 环境动力学
    '''模拟小型方格世界的环境动力学特征
    Args:
        s 当前状态 int 0 - 15
        a 行为 str in ['n','e','s','w'] 分别表示北、东、南、西
    Returns: tuple (s_prime, reward, is_end)
        s_prime 后续状态
        reward 奖励值
        is_end 是否进入终止状态
    '''
    s_prime = s
    if ( (s==1 or s == 5 or s==8 or s == 7) and a == "w") or (s<=4 and a == "n") \
        or ( (s==4 or s == 7 or s == 10 or s == 6) and a == "e") or ((s == 5 or s==8 or s== 10) and a == "s")\
        or (s == 9 and a == 'n')\
        or s in [1, 11]:
        pass
    else:
        s_prime
while delta < theta:
    for num in range(len(S)):
        # print(state_Num)
        v = V[num]
        for i in range(4):

