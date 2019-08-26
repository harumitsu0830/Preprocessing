#フラグ作成
#条件分岐の関数を定義
def func_make_flag(x):
    if x<=:
        return 0
    elif  < x <= :
        return 1
    else:
        return 2

#データフレームの指定
DF = 

DF.loc[:,'_flag'] = DF.loc[:,''].apply(func_make_flag).values
