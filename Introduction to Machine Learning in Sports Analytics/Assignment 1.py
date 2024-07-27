# Let's first make sure all of our columns are numeric for col in observations.columns:
for col in observations.columns:
    if col != 'outcome_categorical':
        observations [col]=pd.to_numeric(observations [col])

observations-observations.dropna (subset=["away_cap", "home_cap"])

# Let's put the first 800 observations in our training data. 
training_df=observations [0:799] 
testing_df=observations [800:]

# And now lets impute the missing data for each set independently 
training_df=training_df.fillna(training_df.mean()) 
testing_df=testing_df.fillna(testing_df.mean())