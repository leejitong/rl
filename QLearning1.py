#QLearning
import numpy as np
import pandas as pd
import time#移動速度

np.random.seed(2)#原来每次运行代码时设置相同的seed，则每次生成的随机数也相同。

N_STATES = 6
ACTIONS = ['left','right']
EPSILON = 0.9 #百分之90的幾率選擇最優動作（控制貪婪程度）
ALPHA = 0.1 #學習效率
LAMBDA = 0.9#衰減率
MAX_EPISODES = 13#回合次數
FRESH_TIME = 0.5#每個動作的秒數方便觀察

def build_q_table(n_states,actions):
    table = pd.DataFrame(
            np.zeros((n_states,len(actions))),#初始化q—table全0
            columns = actions,#設置欄位名稱（列名稱）為left和right
            )
    print(table)
    return table

#build_q_table(N_STATES,ACTIONS)
    
def choose_action(state, q_table):#根據狀態和q—table中的值選動作
    state_actions = q_table.iloc[state, :]#選出該STATE下所有的動作值
    if (np.random.uniform() > EPSILON) or (state_actions.all() == 0): #非贪婪or或者这个state还没有探索过(探索過程)
        action_name = np.random.choice(ACTIONS)
    else:
        action_name = state_actions.argmax()#贪婪模式(選擇最大值)
    return action_name

def get_env_feedback(S, A):#環境反饋
    if A == 'right':#向右
        if S == N_STATES - 2:#因爲N_STATES從0開始不是從1開始所以減2
            S_ = 'terminal'#終點(拿到獎勵)
            R = 1
        else:
            S_ = S + 1
            R = 0
    else:#向左
        R = 0
        if S == 0:
            S_ = S 
        else:
            S_ = S - 1
    return S_, R

#環境
def update_env(S, episode, step_counter):
    # This is how environment be updated
    env_list = ['-']*(N_STATES-1) + ['T']   # '---------T' our environment
    if S == 'terminal':
        interaction = 'Episode %s: total_steps = %s' % (episode+1, step_counter)
        print('\r{}'.format(interaction), end='')
        time.sleep(2)
        print('\r                                ', end='')
    else:
        env_list[S] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(FRESH_TIME)

#核心        
def rl():
    q_table = build_q_table(N_STATES, ACTIONS)  # 初始 q table
    for episode in range(MAX_EPISODES):     # 回合
        step_counter = 0#步數
        S = 0   # 回合初始位置
        is_terminated = False   # 是否回合结束
        update_env(S, episode, step_counter)    # 初环境更新
        while not is_terminated:

            A = choose_action(S, q_table)   # 选行为
            S_, R = get_env_feedback(S, A)  # 实施行为并得到环境的反馈
            q_predict = q_table.loc[S, A]    # 估算的(状态-行为)值
            if S_ != 'terminal':
                q_target = R + LAMBDA * q_table.iloc[S_, :].max()   #  实际的(状态-行为)值 (回合没结束)
            else:
                q_target = R     #  实际的(状态-行为)值 (回合结束)
                is_terminated = True    # terminate this episode

            q_table.loc[S, A] += ALPHA * (q_target - q_predict)  #  q_table 更新
            S = S_  # 探索者移动到下一个 state

            update_env(S, episode, step_counter+1)  # 环境更新

            step_counter += 1
    return q_table

if __name__ == "__main__":
    q_table = rl()
    print('\r\nQ-table:\n')
    print(q_table)






