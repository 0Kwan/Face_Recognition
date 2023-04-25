import pandas as pd 

#test를 위해선 list의 txt파일이 필요로 하여 생성하는 코드


df = pd.read_excel('./data/AI_DATA_0403_1.xlsx', sheet_name='Sheet1', header=1)

for i, r in df['profiles'].items():
    result = r.split(',')
    df.loc[i, 'profile1'] = result[0]
    if len(result) >= 2:
        df.loc[i, 'profile2'] = result[1]
    if len(result) == 3:
        df.loc[i, 'profile3'] = result[2]
with open("result_values.txt", "w") as f:
    for idx, row in df.iterrows():
        folder_name = str(row['id'])
        for i in range(1, 4):
            if f"profile{i}" in row and pd.notnull(row[f"profile{i}"]): # 수정된 부분입니다
                profile_path = f"images/{folder_name}/{row['id']}_{i}.jpg"
                auth_path = f"images/{folder_name}/{row['id']}_auth.jpg"
                if row['labeled_result_value(0: 미인증, 2: 인증)'] == 2:
                    result_value = 1
                else :
                    result_value = 0
                f.write(f"{auth_path} {profile_path} {result_value}\n")