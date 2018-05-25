# depression_in_text
This is a logistic regression model trained on tweets that use the word "depression" vs. random tweets in order to learn what words, other than "depression", "depressed", etc, are indicators of someone talking about depression which may be a good proxy for someone that is depressed.

## How to use this
1. Install the anaconda release of 3.6 Python (this will have most of the packages the model uses) - https://www.anaconda.com/download/#macos
2. Clone this git repo:
- open terminal on your computer
- get the clone link from the top right of this page
- in terminal type the command `git clone the_link_at_the_top_right_of_page`
3. In terminal type the command `jupyter notebook`
- This should open a browser window with the files listed at the top of this page
4. Click on Predict Depressed Tweet vs. Random Tweet.ipynb
5. A new window will open. From the cell menu at the top click "Run All"
6. The last cell at the bottom of the page will allow one to create their own journal entry, score it and see how each word contributed to the score
