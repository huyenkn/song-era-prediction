# song-era-prediction-nlp

I implemented script to build a system that automatically classifies song lyrics by era.

Performance is measured by accuracy (percentage of era labels that are correctly predicted). I applied three methods: Naive Bayes, Perceptron, and Logistic regression. The best method, Logistic regression, achieves 54% in accuracy.

### Data format
The training and validation sets are Pandas DataFrames (read from CSV files, whose shapes are (4000, 2) and (450, 2) respectively). The first column is 'Era' (label) and the second column is 'Lyrics' (a text of song lyric).
