from sklearn.metrics import plot_confusion_matrix
matrix=plot_confusion_matrix(clf, df_pitches[features],df_pitches["pitch_type"], xticks_rotation='vertical', cmap='cividis')
matrix.figure_.set_size_inches(8,8)

from sklearn.metrics import recall_score
pd.DataFrame([recall_score(df_pitches["pitch_type"],clf.predict(df_pitches[features]),average=None)],columns=sorted(df_pitches["pitch_type"].unique()))

from sklearn.metrics import precision_score
pd.DataFrame([precision_score(df_pitches["pitch_type"],clf.predict(df_pitches[features]),average=None)],columns=sorted(df_pitches["pitch_type"].unique()))