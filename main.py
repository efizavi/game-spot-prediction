from bs4 import BeautifulSoup
from bs4 import Comment
import numpy as np
import pandas as pd
import requests
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def load_soup_object(html_file_name):
    with open(html_file_name, encoding='utf8') as infile:
        return BeautifulSoup(infile, "html.parser")

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = ''):
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filledLength = int(100 * iteration // total)
    bar = ' ' * filledLength + '-' * (100 - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = '\r')
    if iteration == total: 
        print()
        
def load_gamespot_dataframe(page_count):
    df = pd.DataFrame(columns=['name', 'release_date', 'platforms', 'user_avg', 'score', 'score_word'])
    names = []
    release_dates = []
    platforms = []
    user_avgs = []
    scores = []
    score_words = []
    
    for page in range(1, page_count+1):
        soup = load_soup_object(str(page) + '.html')
        for game_soup in soup.findAll('div', attrs={'class': 'media-game'}):
            h3_name = game_soup.find('h3', attrs={'class': 'media-title'})
            if h3_name:
                names.append(h3_name.string.strip())
            else:
                names.append(np.nan)
                
            time = game_soup.find('time', attrs={'class': 'media-date'})
            if time:
                release_dates.append(time['datetime'])
            else:
                release_dates.append(np.nan)
                
            systems = game_soup.find('ul', attrs={'class': 'system-list'})
            if systems:
                scraped_plats = ''
                for li in systems.find_all('li'):
                    if ('system--pill' in li['class']):
                        scraped_plats += (li.findAll('span')[0].string.strip()) + ','
                platforms.append(scraped_plats[:len(scraped_plats) - 1])
            else:
                platforms.append(np.nan)
            
            user_review = game_soup.find('div', attrs={'class': 'media-well--review-user'})
            if user_review:
                user_review_content = user_review.find('strong')
                if (user_review_content):
                    user_avgs.append(user_review_content.string.strip())
                else:
                    user_avgs.append(np.nan)
            else:
                user_avgs.append(np.nan)
            
            gs_review = game_soup.find('div', attrs={'class': 'media-well--review-gs'})
            if gs_review:
                score = gs_review.find('span').find('strong')
                if score:
                    scores.append(score.string.strip())
                else:
                    scores.append(np.nan)
                score_word = gs_review.findAll('span')[1]
                if score_word:
                    score_words.append(score_word.string.strip())
                else:
                    score_words.append(np.nan)
            else:
                scores.append(np.nan)
                score_words.append(np.nan)
        printProgressBar(page, page_count, prefix = 'Progress:', suffix = 'complete')
    
    df['name'] = names
    df['release_date'] = pd.to_datetime(release_dates)
    df['platforms'] = platforms
    df['user_avg'] = user_avgs
    df['score'] = scores
    df['score_word'] = score_words
    return df

def load_dataframe(file_name):
    return pd.read_csv(file_name)

def replace_ambigous_chars(col_name, df):
    df[col_name].replace({'-':''}, regex=True, inplace=True)
    df[col_name].replace({':':''}, regex=True, inplace=True)
    df[col_name].replace({"'":''}, regex=True, inplace=True)
    df[col_name].replace({'\(':''}, regex=True, inplace=True)
    df[col_name].replace({'\)':''}, regex=True, inplace=True)

def transfer_to_categorical(df, categorical_col_names):  
    df_cpy = df.copy()
    df_cpy = pd.get_dummies(df_cpy, columns=categorical_col_names, prefix=categorical_col_names)
    return df_cpy

def get_frequent_elements(df, col_name, num_top_elements):
    return df[col_name].value_counts().head(num_top_elements).sort_index()

def one_dim_plot(sr, plot_type, axis):
    return sr.plot(kind=plot_type, ax=axis)

def get_highly_correlated_cols(df, min_thresh, max_thresh):
    correlations = []
    tuple_arr = []
    corr = df.corr()
    for i in range(len(corr)):
        for j in range(i+1, len(corr)):
            if corr.iloc[i,j] >= min_thresh and corr.iloc[i,j] <= max_thresh:
                correlations.append(corr.iloc[i,j])
                tuple_arr.append((i,j))
    return correlations, tuple_arr

def plot_frequent_elements(df, df_in_params):
    fig, axes = plt.subplots(1,len(df_in_params), figsize=(20,5))
    for i in range(len(df_in_params)):
        curr_col = df_in_params['col_name'][i]
        curr_plot_type = df_in_params['plot_type'][i]
        curr_num_top_elem = df_in_params['num_top_elements'][i]
        
        curr_elements = get_frequent_elements(df, curr_col, curr_num_top_elem)
        one_dim_plot(curr_elements, curr_plot_type, axes[i])
        
def cross_tabulation(df, col_name, other_col_name):
    return pd.crosstab(df[col_name],df[other_col_name],normalize='index')

def plot_cross_tabulation(df, col_name, other_col_name):
    fig, axes = plt.subplots(1, 1, figsize=(25,5))
    ct = cross_tabulation(df, col_name, other_col_name)
    ct.plot(kind='bar',ax=axes)

def plot_scatters(df, min_thresh, max_thresh):
    correlations, tuple_arr = get_highly_correlated_cols(df, min_thresh, max_thresh)
    fig, axes = plt.subplots(1,len(correlations), figsize=(20,5))
    for i in range(len(correlations)):
        if len(correlations) > 1:
            ax = axes[i]
        else:
            ax = axes
        ax.set_title("corr('%s', '%s')=%4.2f" %(df.columns[tuple_arr[i][0]], df.columns[tuple_arr[i][1]], correlations[i]) )
        ax.scatter(df[df.keys()[tuple_arr[i][0]]], df[df.keys()[tuple_arr[i][1]]])



# We used this code to have our data available offline

def html_to_bs(url):
    page = requests.get(url)
    return BeautifulSoup(page.content, 'html.parser')

def write_file(soup, name):
    [x.extract() for x in soup.find_all('script')]
    [x.extract() for x in soup.find_all('style')]
    [x.extract() for x in soup.find_all('meta')]
    [x.extract() for x in soup.find_all('noscript')]
    [x.extract() for x in soup.find_all(text=lambda text:isinstance(text, Comment))]
    html = soup.prettify("utf-8")
    with open(name, "wb") as file:
        file.write(html)

def write_gamespot_html_files():
    print('Scraping gamespot.com website...')
    for page in range(1,439):
        soup = html_to_bs('https://www.gamespot.com/new-games/?sort=score&game_filter%5Bplatform%5D=all&game_filter%5BminRating%5D=1&game_filter%5BtimeFrame%5D=all&game_filter%5BstartDate%5D=&game_filter%5BendDate%5D=&game_filter%5Btheme%5D=&game_filter%5Bregion%5D=&page='+str(page))
        write_file(soup, str(page) + '.html')
        print(page/438*100)

# Remove duplicate rows from dataframe
print('Loading scraped HTML files...')
df = load_gamespot_dataframe(438)
print('GameSpot DataFrame loaded.')

print('Cleaning bad data...')
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

print('Loading secondary DataFrame')
df_sales = load_dataframe('vgsales.csv')
print('Videogame Sales DataFrame loaded.')
print('Removing duplicates...')
df_sales.drop_duplicates(inplace=True)

# Prepare new columns for our main dataframe
df['genre'] = 'unknown'
df['publisher'] = 'unknown'
df['na_sales'] = np.nan
df['eu_sales'] = np.nan
df['jp_sales'] = np.nan
df['other_sales'] = np.nan
df['global_sales'] = np.nan

append_counter = 0
progress_counter = 0

# Clean up ambigous charachters from video game names to improve datasets correlation
print('Formatting data...')
replace_ambigous_chars('name', df)
replace_ambigous_chars('Name', df_sales)

print('Merging Videogame Sales data to GameSpot dataframe...')
for row in df_sales.iterrows():
    cond = df[df['name'] == row[1]['Name']]
    if len(cond) > 0:
        append_counter += 1
        df.at[int(cond.index[0]), 'genre'] = row[1]['Genre']
        df.at[int(cond.index[0]), 'publisher'] = row[1]['Publisher']
        
        if np.isnan(df.at[int(cond.index[0]), 'na_sales']):
            df.at[int(cond.index[0]), 'na_sales'] = 0.0
            df.at[int(cond.index[0]), 'eu_sales'] = 0.0
            df.at[int(cond.index[0]), 'jp_sales'] = 0.0
            df.at[int(cond.index[0]), 'other_sales'] = 0.0
            df.at[int(cond.index[0]), 'global_sales'] = 0.0
            
        df.at[int(cond.index[0]), 'na_sales'] += row[1]['NA_Sales']
        df.at[int(cond.index[0]), 'eu_sales'] += row[1]['EU_Sales']
        df.at[int(cond.index[0]), 'jp_sales'] += row[1]['JP_Sales']
        df.at[int(cond.index[0]), 'other_sales'] += row[1]['Other_Sales']
        df.at[int(cond.index[0]), 'global_sales'] += row[1]['Global_Sales']
    progress_counter += 1
    printProgressBar(progress_counter, len(df_sales)-1)
    
print('Appended to ' + str(append_counter) + 'rows')

print('Filling missing data...')
replace_vals = {'na_sales': df_sales.NA_Sales.median(),\
                'eu_sales': df_sales.EU_Sales.median(),\
                'jp_sales': df_sales.JP_Sales.median(),\
                'other_sales': df_sales.Other_Sales.median(),\
                'global_sales': df_sales.Global_Sales.median()}

df.fillna(value=replace_vals, inplace=True)

# We used the following code to learn more about our dataframe correlations and plot our graphs:
    
# ------------------------------------------------------------------------------------------------

# df_params = pd.DataFrame({'plot_type': ['line', 'bar', 'pie'], 
#                           'col_name': ['release_date', 'publisher', 'genre'],
#                           'num_top_elements': [30,12,10]})

#df_without_unknown = df.copy()
#df_without_unknown = df_without_unknown.drop\
    #(df_without_unknown[df_without_unknown.publisher == 'unknown'].index)
#plot_frequent_elements(df_without_unknown, df_params)

#freq_pub = get_frequent_elements(df_without_unknown, 'publisher', 12)
#df_without_unknown = df_without_unknown[[x in freq_pub for x in df_without_unknown.publisher]]
#df_without_unknown.dropna(subset=['publisher'], inplace=True

# Convert platform lists to categorical values
# all_platforms = []
# [all_platforms.extend(plat.split(',')) for plat in df_without_unknown.platforms]
# for unique_plat in np.unique(all_platforms):
#     if len(unique_plat) > 0:
#         df_without_unknown['is_'+str(unique_plat)] = ([int(unique_plat in plat) for plat in df_without_unknown.platforms])

# Which genres produce the best scoring games?

#plot_cross_tabulation(df_without_unknown, 'genre', 'score_word')
#plot_cross_tabulation(df_without_unknown, 'publisher', 'score_word')
#df_without_unknown = transfer_to_categorical(df_without_unknown, ['publisher'])


# Here we experimented with scatter plots of different columns
#plot_scatters(df_without_unknown[['global_sales', 'genre', 'user_avg']], 0.8, 0.99)
# ------------------------------------------------------------------------------------------------

# Add a platform count row for each game
print('Editing data for prediction assistance...')
plat_count = []
for index, row in df.iterrows():
    plat_count.append(len(row.platforms.split(',')))
df['platform_count'] = plat_count

# Categorial columns to numerical
all_platforms = []
[all_platforms.extend(plat.split(',')) for plat in df.platforms]
for unique_plat in np.unique(all_platforms):
    if len(unique_plat) > 0:
        df['is_'+str(unique_plat)] = ([int(unique_plat in plat) for plat in df.platforms])
     
# Drop columns which won't help us predict
df.drop('platforms', axis='columns', inplace=True)
df.drop('score_word', axis='columns', inplace=True)
df.drop('name', axis='columns', inplace=True)

# Format columns to numerical data    
df['release_date'] = [date.year for date in df['release_date']]
df['genre'] = df[['genre']].apply(lambda x: x.astype('category').cat.codes)
df['publisher'] = df[['publisher']].apply(lambda x: x.astype('category').cat.codes)

#Split to train and test
print('Predicting...')
X = df.drop(['score'], axis='columns')
y = df['score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=41)

# Predict with Linear Regression:
lr = LinearRegression().fit(X_train, y_train)
df_pred = pd.DataFrame({'Actual': y_test.array, 'Predicted':lr.predict(X_test)})

print("Prediction accuracy:")
print((df_pred['Predicted'].astype(float).mean()/df_pred['Actual'].astype(float).mean()))

print("Plotting prediction")
df_pred.Actual = df_pred.Actual.astype(float)
df_pred.head(100).plot()


print('All done!')