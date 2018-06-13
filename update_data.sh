#ssh into pi
ssh -p 1722 pi@shremore.chickenkiller.com

#cd into depression in text
cd depression_id_text

#git pull origin master
git pull origin master

#archive old data sets into archive foler
DATE=`date +%Y-%m-%d`
file_name='depression_tweets'
file_ext='.csv'
mv  $file_name$file_ext archive/$file_name$DATE$file_ext

#concat all files into one a place in depression_id_text folder as depression_tweets.csv
cat ~/cronjobs/twitter_data_depression/tweets/*.csv > $file_name$file_ext

#git push origin master
git push origin master

#count number of lines in depression_tweets.csv save as variable

#archive old random tweets.csv

#kick of python script to collect as many random tweets as variable just defined