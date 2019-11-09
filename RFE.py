def RFE(DF, EXECUTIONS, COLS_TO_DEL, TARGET, SMOTE_RATE, CUT):
    '''
    DF:dataframe
    EXECUTIONS:実行サイクル数
    COLS_TO_DEL：初期段階で削除するカラム
    TARGET：目的変数
    SMOTE_RATE：データ数最大のラベルに対する他ラベルの比率
    CUT：1サイクルで削る説明変数の個数
    '''

    for i in range(EXECUTIONS ):
        print(i)
        # データ整形工程。SMOTEで不均衡を改善する。
        #########################################################################################################
        DF_ = DF.dropna(how='any',axis=0)

        COLS_TO_DEL_ = COLS_TO_DEL
        DF_ = DF_.drop(COLS_TO_DEL_,axis=1)

        X = DF_.copy().drop(TARGET,axis=1)
        X = X.values

        y = DF_[TARGET].values
        le = LabelEncoder()
        y = le.fit_transform(y)

        # 訓練、テスト分割
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        # SMOTE
        dic =  {}
        for i in np.unique(y_train):
            dic[i] = np.count_nonzero(y_train == i)

        dic_resumpled = {}
        for i in np.unique(y_train):
            if np.count_nonzero(y_train == i) != max(dic.values()):
                dic_resumpled[i] = int(round(max(dic.values()) * SMOTE_RATE ,0))
            else:
                dic_resumpled[i] = max(dic.values())

        smote = SMOTE('not majority',ratio=dic_resumpled)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

        n_train = int(X_train_resampled.shape[0] * 0.75)
        X_train_2, X_val = X_train_resampled[:n_train], X_train_resampled[n_train:]
        y_train_2, y_val = y_train_resampled[:n_train], y_train_resampled[n_train:]

        # 標準化
        scaler = StandardScaler()
        X_train_2 = scaler.fit_transform(X_train_2)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

        unique_elements, counts_elements = np.unique(y_train_resampled, return_counts=True)
        print("Frequency of unique values of the said array:")
        print(np.asarray((unique_elements, counts_elements)))

        #ligthgbm
        #########################################################################################################
         # 目的関数
        def objective(trial):

            num_leaves = int(trial.suggest_discrete_uniform("num_leaves", 6, 31, 5))
            min_data_in_leaf = int(trial.suggest_discrete_uniform("min_data_in_leaf", 1, 5, 1))
            max_depth = int(trial.suggest_discrete_uniform("max_depth", 1, 5, 1))
            max_bin = int(trial.suggest_discrete_uniform("max_bin", 50, 250, 50))



            gbm = lgb.LGBMClassifier(objective='multiclass',
                                     num_class = 3,
                                     num_leaves = num_leaves,
                                     min_data_in_leaf = min_data_in_leaf,
                                     max_depth = max_depth,
                                     max_bin = max_bin,
                                     learning_rate=0.1,
                                     min_child_samples=10,
                                     n_estimators=100)

            gbm.fit(X_train, y_train)
            return 1.0 - accuracy_score(y_test, gbm.predict(X_test))

        # optuna
        study = optuna.create_study()
        study.optimize(objective, n_trials=100)

        # 最適解
        print(study.best_params)
        print(study.best_value)
        print(study.best_trial)

        ### RFのパラメータ名をint型にする
        params=study.best_params.copy()
        for _k in['num_leaves','min_data_in_leaf','max_depth','max_bin']: params[_k] = int(params[_k])
        print(params)

        # チューニングしたモデルの精度・特徴量確認
        #########################################################################################################

        DF_ = DF.dropna(how='any',axis=0)
        DF_ = DF_.drop(cols_to_del,axis=1)

        X = DF_.copy().drop(TARGET,axis=1)
        X = X.values

        y = DF_[TARGET].values
        le = LabelEncoder()
        y = le.fit_transform(y)

        # 訓練、テスト分割
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        # SMOTE
        smote = SMOTE(ratio=dic_resumpled)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

        # 標準化
        scaler = StandardScaler()
        X_train_resampled = scaler.fit_transform(X_train_resampled)
        X_test = scaler.transform(X_test)

        # gbm
        gbm = lgb.LGBMClassifier(objective='multiclass',
                                 num_class = 3,
                                 learning_rate=0.1,
                                 min_child_samples=10,
                                 n_estimators=100,
                                **params)

        gbm.fit(X_train_resampled,  y_train_resampled)

        print("トレーニングデータ: ", gbm.score(X_train_resampled, y_train_resampled) )
        print("テストデータ: ", gbm.score(X_test,y_test))

        #混合マトリックス
        model = gbm

        # 学習データのConfusionMatrix
        pred_train_resampled = model.predict(X_train_resampled)
        confusion = confusion_matrix(y_train_resampled,pred_train_resampled)
        print("学習データのConfusion matrix:\n{}".format(confusion))

        print('\n-----------------------------------\n')

        # 検証データのConfusionMatrix
        pred_test = model.predict(X_test)
        confusion = confusion_matrix(y_test,pred_test)
        print("テストデータのConfusion matrix:\n{}".format(confusion))

        feat_labels = DF.columns

        # 特徴量の重要度
        importances = gbm.feature_importances_
        # 重要度の降順で特徴量のインデックスを抽出
        indices = np.argsort(importances)[::-1]
        #重要度の降順で特徴量の名称、重要度を表示
        for f in range(X_train_resampled.shape[1]):
            print('%2d) %-*s %.3f' % (f+1,30,feat_labels[indices[f]], importances[indices[f]]))

        # 下位CUT個を消去対象とする。
        cols = []

        for f in range(X_train_resampled.shape[1]):
            cols.append(feat_labels[indices[f]])

        COLS_TO_DEL_ += cols[-CUT:]
        print('cols_to_del',cols_to_del)
        print('-----------------------------------------------------------------------------------------------------------------------------------------------------')
