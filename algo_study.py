#!/usr/bin/env python
# coding: utf-8

# ## 프로그래밍 대회에서 배우는 알고리즘 문제해결전략 1
# ## algospot

# ### 1 Festival

# In[25]:


'''
def input_hendler(minimum, maximum, value) :
    if(minimum <= value and value < maximum) :
        print("input error : {}".format(value))
        return False
    return True
'''
# 입력 받는 함수
def Input(c,n,l,n_cost) :
    # case number 입력
    c = int(input())
    for num in range(0,c) :
        # n & l 입력
        n_l = input()
        # n분리
        n.append(int(n_l.split(" ")[0]))
        #l분리
        l.append(int(n_l.split(" ")[1])) 
        # day_n_cost 입력
        days_n_cost = input()
        #여백 제거
        days_n_cost = days_n_cost.split(" ")
        n_cost.append(list())
        #n_cost에 day_n_cost int로 바꿔서 저장
        for day_cost in days_n_cost :
            n_cost[num].append(int(day_cost))
    return c

# 평균 값 구하고 min_avg 갱신
def get_min_avg(total, num, min_avg) :
    avg = total / num
    return min(min_avg, avg) 

# 
def cal(c,n,l,n_cost) :
    #test case만큼 반복
    for num in range(0,c) :
        min_avg = 1000
        total = 0
        #평균 구하기 반복
        for n_num in range(0, n[num] - l[num] + 1) :
            #합 구하기
            for loop_num in range(0,l[num]) :
                total += n_cost[num][n_num + loop_num]
            #합으로 평균 구하기
            min_avg = get_min_avg(total, num, min_avg)
            #한 개씩 추가하면서 평균구하기 
            for i in range(n_num + l[num], n[num]) :
                total += n_cost[num][i]
                min_avg = get_min_avg(total, i - n_num + 1, avg)
            total = 0
        print(min_avg)

def main() :
    c = 0
    n = list()
    l = list()
    n_cost = list()
    c = Input(c,n,l,n_cost)
    cal(c,n,l,n_cost)
    
    
if __name__ == "__main__":
    main()
    
'''
합의 평균을 barnch & bound로 구한다
시간은 O(n^2)
제한시간이 길어서 풀었지만 제한시간이 짧았다면 글쎄........
다른 정답 봤는데 나랑 코드 비슷하던데 왜 시간차이가....?
'''


# ### 2 문제 순서는 난이도 순이 아닙니다

# In[8]:


#입력 받기
def prb_hard_input(problem_num, hards, case) :
    problem_num.append(int(input()))
    hards.append(input())
    hards[case] = hards[case].split(" ")

# 난이도 순서로 정렬 되어있는지 확인하고 수 세기
def check_hard_sorting_add(hard_sort, hards, index, case) :
    if(int(hards[case][index]) == (index + 1)) :
                return hard_sort + 1
    return hard_sort

def main() :
    case_num = int(input())
    problem_num = list()
    hards = list()
    hard_sort = 0
    for case in range(0, case_num) :
        prb_hard_input(problem_num, hards, case)
        
    for case in range(0, case_num) :
        for index in range(0, problem_num[case]) :
            hard_sort = check_hard_sorting_add(hard_sort, hards, index, case)
        print(hard_sort)
        hard_sort = 0
    
'''
2차원 해야하는데 1차원으로하고 1차원은 0차원으로 해서 삽질했었음
'''
    
if __name__ == "__main__" :
    main()


# ### 3 PICNICK

# In[33]:


# 행렬 초기화
def friend_mat_init(friend_mat, student_num) :
    for n in range(0, student_num) :
        friend_mat.append(list())
        for m in range(0, n) :
            friend_mat[n].append(0)

# 짝 지어주기
import copy
def search_cup(friend_mat, index_set) :
    result = 0
    # 학생들을 모두 짝지었다면 1추가
    if(len(index_set) == 0) :
        return 1
    col = index_set[0]
    #남은 학생들 짝 지어주기
    for row in index_set :
        # 배열 크기 벗어나지 않게 예외 처리
        if(len(friend_mat[row]) > col) :
            # 짝 지어줄 수 있다면
            if(friend_mat[row][col] == 1) :
                my_index_set = copy.deepcopy(index_set)
                # 짝 지어준 학생 삭제
                my_index_set.remove(row)
                my_index_set.remove(col)
                # 결과 추가
                result += search_cup(friend_mat, my_index_set)
    return result

def test():
    case_num = int(input())
    student_num = list()
    friend_num = list()
    friendship = list()
    for case in range(0, case_num) :
        st_fr = input()
        st_fr = st_fr.split(" ")
        student_num.append(int(st_fr[0]))
        friend_num.append(int(st_fr[1]))
        friendship_set = input()
        friendship_set = friendship_set.split(" ")
        temp_ship_set = list()
        for n in range(0, friend_num[case]) :
            temp_ship = list()
            temp_ship.append(int(friendship_set[2 * n]))
            temp_ship.append(int(friendship_set[2 * n + 1]))
            temp_ship_set.append(temp_ship)
        friendship.append(temp_ship_set)
        
    
    for case in range(0, case_num) :
        friend_mat = list()
        friend_mat_init(friend_mat, student_num[case])
        index_set = list()
        for student in range(0, student_num[case]) :
            index_set.append(student)
        for friend in range(0, friend_num[case]) :
            row = max(friendship[case][friend][0],friendship[case][friend][1])
            col = min(friendship[case][friend][0],friendship[case][friend][1])
            friend_mat[row][col] = 1
        result = search_cup(friend_mat, index_set)
        print(result)

if __name__ == '__main__' :
    test()


# ### 4 에어컨을 끈다고 전력난이 해결될까?

# In[3]:


# input 받은거 int로 고쳐야 하는거 까먹지 말자

case_num = int(input())
goal_ele = list()
hour_ele = list()

for case in range(0, case_num) :
    goal_ele.append(int(input()))
    hour_ele.append(input().split(" "))
    
for case in range(0, case_num) :
    total_ele = 0
    for hour in range(0, len(hour_ele[case])) :
        total_ele += int(hour_ele[case][hour])
    if(total_ele <= goal_ele[case]) :
        print('YES')
    else :
        print('NO')
        


# ### 5 BOGGLE

# In[36]:


#재귀문 시간 초과
#동적 계획법
#경우의 수 2가지
#단어를 만들거나 못 만들거나
#단어를 못 만들면 0
#단어를 만들면 1
#초기화 -1
#재귀함수로 구현
#재귀문에서 판단
#리턴 중 최대값 캐시에 저장

MAX_WORD_LENGTH = 10
MAP_SIZE = 5
cache = list()

def init_cache() :
    global cache
    cache = list()
    for n in range(0, MAX_WORD_LENGTH) :
        cache.append(list())
        for row in range(0, MAP_SIZE) :
            cache[n].append(list())
            for col in range(0, MAP_SIZE) :
                cache[n][row].append(-1)
                
def search_letter(boggle_map, start_row, end_row, start_col, end_col, word, word_num, pos) :
    global cache

    if(word_num == len(word)) :
        return 1
    result = 0
    for row in range(start_row, end_row) :
        for col in range(start_col, end_col) :
            cur = [row, col]
            if(cache[word_num][row][col] != 0 and cur != pos):
                if(boggle_map[row][col] == word[word_num] and cur != pos) :
                    pos = [row, col]
                    scope = get_scope(pos)
                    result = search_letter(boggle_map, scope[0], scope[1], scope[2], scope[3], word, word_num + 1, pos)
                cache[word_num][row][col] = result
                if(result == 1) :
                    return 1
    return 0

def get_scope(pos) :
    start_row = max(pos[0] - 1, 0)
    end_row = min(pos[0] + 2, MAP_SIZE)
    start_col = max(pos[1] - 1, 0)
    end_col = min(pos[1] + 2, MAP_SIZE)
    scope = [start_row, end_row, start_col, end_col]
    return scope

def print_res(word, success) :
    if(success == True) :
        print("{} YES".format(word))
    else :
        print("{} NO".format(word))
        
def boggle() :
    case_num = int(input())
    multi_boggle_map = list()
    multi_word = list()
    for case in range(0, case_num) :
        temp_boggle_map = list()
        for row in range(0, MAP_SIZE) :
            temp_boggle_map.append(input())
        #temp_boggle_map = [['U','R','L','P','M'],['X','P','R','E','T'],['G','I','A','E','T'],['X','T','N','Z','Y'],['X','O','Q','R','S']]
        multi_boggle_map.append(temp_boggle_map)
        word_num = int(input())
        temp_words = list()
        for num in range(0, word_num) :
            temp_words.append(input())
        multi_word.append(temp_words)
    
    for case in range(0, case_num) :
        boggle_map = multi_boggle_map[case]
        words = multi_word[case]
        for word in words :
            init_cache()
            pos = [-1, -1]
            success = search_letter(boggle_map, 0, MAP_SIZE, 0, MAP_SIZE, word, 0, pos)
            print_res(word, success)



def main() :
    boggle()
    
    
if __name__ == '__main__' :
    main()


# ### 6 BORDARCORVER

# In[6]:


#런타임 오류 : 배열 범위 벗어나는 핸들러 작성 미흡해서 났었음

def main() :
    case_num = int(input())
    multi_height = list()
    multi_length = list()
    multi_board = list()
    for case in range(0, case_num) :
        h_w = input()
        multi_height.append(int(h_w.split(" ")[0]))
        multi_length.append(int(h_w.split(" ")[1]))
        input_board = list()
        for height in range(0, multi_height[case]) :
            input_board.append(input())
        multi_board.append(input_board)
        
    for case in range(0, case_num) :
        height = multi_height[case]
        length = multi_length[case]
        board = multi_board[case]
        result = put_block(board)
        print(result)
    
    
def search_position(board, height, length) :
    for row in range(0, height) :
        for col in range(0, length) :
            if(board[row][col] == '.') :
                pos =[row, col]
                return pos
    return -1

def is_in(x, y, board) :
    if(0 <= x and x < len(board[0]) and 0 <= y and y < len(board)) :
        return True
    return False

def letter_switch(board, row, col, dy, dx, next_dy, next_dx, letter) :
    board[row] = board[row][:col] + letter + board[row][col + 1:]
    board[dy] = board[dy][:dx] + letter + board[dy][dx + 1:]
    board[next_dy] = board[next_dy][:next_dx] + letter + board[next_dy][next_dx + 1:]

def put_block(board) :
    block_y = [1,1,0,1,1,1,0,1]
    block_x = [0,1,1,1,0,-1,1,0]
    res = 0
    start = search_position(board, len(board), len(board[0]))
    if(start is -1) :
        return 1
    row = start[0]
    col = start[1]
    for block_no in range(0,4) :
        pos = block_no * 2
        dy = row + block_y[pos]
        dx = col + block_x[pos]
        next_dy = row + block_y[pos + 1]
        next_dx = col + block_x[pos + 1]
        if(is_in(dx, dy, board) == True and is_in(next_dx, next_dy, board) == True) :
            if(board[dy][dx] == '.' and board[next_dy][next_dx] == '.') :
                letter_switch(board, row, col, dy, dx, next_dy, next_dx, '#')
                res += put_block(board)
                letter_switch(board, row, col, dy, dx, next_dy, next_dx, '.')    
    return res

if __name__ == '__main__' :
    main()
                


# ### 7 Sum

# In[53]:


#for문과 재귀함수와 DQ
def main() :
    number = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    print('for문 : ',for_sum(number))
    print('재귀함수 : ', recursive_sum(number))
    print('Divide_Conquer_sum : ', d_q_sum(number))

def for_sum(number) :
    sum = 0
    for index in range(0, len(number)) :
        sum += number[index]
    return sum

def recursive_sum(number) :
    if(len(number) == 1) :
        return number[0]
    return recursive_sum(number[:len(number) - 1]) + number[len(number) - 1]

def d_q_sum(number) :
    if(len(number) == 1) :
        return number[0]
    return d_q_sum(number[:int(len(number) /2)]) + d_q_sum(number[int(len(number) /2) :])

if __name__ == '__main__' :
    main()


# ### 8 QUADTREE

# In[1]:


def find_squar(letter, start) :
    if(letter[start] == 'x') :
        x_letter = 'x'
        left = list()
        right = list()
        left.append(find_squar(letter, start + 1))
        left.append(find_squar(letter, left[0][1] + 1))
        right.append(find_squar(letter, left[1][1] + 1))
        right.append(find_squar(letter, right[0][1] + 1))
        res = x_letter + right[0][2] + right[1][2] + left[0][2] + left[1][2]
        return [left[0][0] - 1, right[1][1], res]
    else :
        return [start, start, letter[start]]
    
def main() :
    case_num = int(input())
    multi_letter = list()
    for case in range(0, case_num) :
        multi_letter.append(input())    
    for case in range(0, case_num) :
        letter = multi_letter[case]
        result = find_squar(letter, 0)
        print(result[2])

              
if __name__ == '__main__' :
    main()


# ### 9 CLOCKSYNC

# In[ ]:


# 머리 안돈다 좀만 쉬자
# 이건 낼 풀어야지
case_num = int(input())

multi_clock = list()
for case in range(0, case_num) :
    multi_clock.appen(input().split(" "))
    
for case in range(0, case_num) :
     
        
def click_switch() :
    click_num = 0
    if(is_clock_12 == True) :
        return 0
    if(len(switch_set) == 0) :
        return 100000000
    for click in range(1, 4) :
        for clock_num in switch :
            clock[clock_num] = (clock[clock_num] + 1) % 4
        prev_click_num = click_num
        click_num = click_num + click_switch(switch_set) + 1
        click_num = min(prev_click_num, click_num)
        
    
    
    


# ### 10 FENCE

# In[26]:


def Input(case_num, multi_line_num, multi_line) :
    case_num = int(input())
    for case in range(0, case_num) :
        multi_line_num.append(int(input()))
        multi_line.append(input().split(" "))
    return case_num

def search_rectangle_version_BF(lines, start, max_area) :
    # 시간 초과
    first_line = int(lines[start])
    length = 1
    for index in range(start + 1, len(lines)) :
        if(int(lines[index]) >= first_line) :
            length += 1
        else :
            break
    for index in range(start - 1, -1) :
        if(int(lines[index]) >= first_line) :
            length += 1
        else :
            break
    area = first_line * length
    max_area = max(max_area, area)
    return max_area
        
def main() :
    case_num = 0
    multi_line_num = list()
    multi_line = list()
    case_num = Input(case_num, multi_line_num, multi_line)
    for case in range(0,case_num) :
        lines = multi_line[case]
        max_area = 0
        for start in range(0, len(lines)) :
            max_area = search_rectangle(lines, start, max_area)
        print(max_area)

    
if __name__ == '__main__' :
    main()
    


# ### 11 JUMP GAME

# In[ ]:


#그래프로 간단하게 풀 수 있다고 하는데 확인해볼 것!!!

#경우의 수 2가지
#범위를 벗어나거나 끝에 도착하던가
#범위를 벗어나면 0
#끝에 도착하면 1
#초기화 -1
#재귀함수로 구현
#재귀문에서 판단
#리턴 중 최대값 캐시에 저장
board = list()
cache = list()
line_num = 0

def Input(multi_line_num, multi_board) :
    case_num = int(input())
    for case in range(0, case_num) :
        line_num = int(input())
        multi_line_num.append(line_num)
        temp_board = list()
        for line in range(0, line_num) :
            row = input().split(" ")
            temp_board.append(row)
        #temp_board = [[2,5,1,6,1,4,1],[6,1,1,2,2,9,3],[7,2,3,2,1,3,1],[1,1,3,1,7,1,2],[4,1,2,3,4,1,3],[3,3,1,2,3,4,1],[1,5,2,9,4,7,0]]
        multi_board.append(temp_board)
    return case_num
def init_cache() :
    global cache
    cache = list()
    for line in range(0, line_num) :
        
        temp_row = list()
        for line in range(0, line_num) :
            temp_row.append(-1)
        cache.append(temp_row)

def search_road(x, y):
    global cache
    if(x >= line_num or y >= line_num) :
        return 0
    delta = int(board[y][x])
    if(delta == 0) :
        return 1
    if(not(cache[y][x] == -1)) :
        return cache[y][x]
    cache[y][x] = max(search_road(x + delta, y), search_road(x, y + delta))
    return cache[y][x]

def main() :
    multi_line_num = list()
    multi_board = list()
    case_num = Input(multi_line_num, multi_board)
    global board
    global line_num
    for case in range(0, case_num) :
        board = multi_board[case]
        line_num = multi_line_num[case]
        init_cache()
        is_goal =search_road(0,0)
        if(is_goal == 1) :
            print('YES')
        else :
            print('NO')

if __name__ == '__main__' :
    main()


# ### 12 WILDCARD

# In[ ]:


def wild_card() :
    if(index >= length) :
        return False
    if(wild_letter != '?' and wild_letter != '*') :
        if(wild_letter == word_letter) :
            return True
        else :
            return False
        
    if(wild_letter == '?') :
        
    if(wild_letter == '*') :


# # 못 푼 문제 9,10

# ### D.Q 
# + #### 계속해서 나누기
# + ####가장 작은 문제 푸는걸 나눈거 마다 반복하기
# 
# ### D.P
# + #### B.F로 풀기
# + #### 반복되는걸 저장해서 계산 축소(memoization)
