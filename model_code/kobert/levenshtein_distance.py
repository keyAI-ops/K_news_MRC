#%%
# 리스트형의 예측값, 정답값 불러오기
def levenshtein_distance(predicts, labels):  # 여기서 predicts, labels input은 list 형태로 들어가야함 ex) predicts = ['ab', 'abc', 'ccc']
    import statistics
    avg_metrics = []
    for num in range(len(predicts)):
        rows, columns = len(predicts[num]) + 1, len(labels[num]) + 1
        # 초기화
        arr = [
            list(range(columns)) if not i else [i]+[0]*(columns-1)
            for i in range(rows)
        ]
        # 거리계산
        for i in range(1, rows):
            for j in range(1, columns):
                if predicts[num][i-1] == labels[num][j-1]:
                    arr[i][j] = arr[i-1][j-1]
                else:
                    arr[i][j] = min(
                        arr[i-1][j], #추가
                        arr[i][j-1], #삭제
                        arr[i-1][j-1], #치환
                    ) + 1
        avg_metrics.append(arr[rows-1][columns-1])    
    result = statistics.fmean(avg_metrics)
    return avg_metrics, result
