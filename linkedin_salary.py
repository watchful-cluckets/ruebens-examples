import pandas as pd
from pandas.io.pickle import read_pickle
import nltk
from nltk import stem, tokenize
from linkedin import linkedin
import numpy as np
import sklearn.cross_validation as cv
from sklearn import linear_model, metrics, ensemble, grid_search, lda, neighbors, tree
import math
import matplotlib.pylab as py
from sklearn.decomposition import PCA
import sklearn.feature_selection as fs
import xlwt

# Linkedin oauth details:
LINKEDIN_CONSUMER_KEY       = ''
LINKEDIN_CONSUMER_SECRET    = ''

# For LinkedIn API calls:
LINKEDIN_OAUTH_USER_TOKEN = ''
LINKEDIN_OAUTH_USER_SECRET = ''
RETURN_URL = 'http://localhost:8000'

stemmer = stem.PorterStemmer()

def estimate_seniority(job_position):
    # Estimates the seniority of a job
    # based on key words in the job position text
    
    # Input
    # job_position: input text
    
    # Output
    # seniority: 'junior', 'default', or 'senior'
    
    seniority = 'default'
    jobtitlewords = job_position.lower().split()
            
    # ignore internships
    if (('intern' in jobtitlewords)
        or ('internship' in jobtitlewords)):
        return 'INTERN'
            
    senior_words = ['sr',
                    'senior',
                    'lead',
                    'principal',
                    'director',
                    'manager',
                    'cto',
                    'chief',
                    'vp',
                    'head'
                    ]
            
    junior_words = ['jr',
                    'junior',
                    'associate',
                    'assistant'
                    ]
            
    for titleword in jobtitlewords:       
                 
        titleword = titleword.replace(',', '')
        titleword = titleword.replace('.', '')  
                      
        if titleword in senior_words:
            seniority = 'senior'
            break
        elif titleword in junior_words:
            seniority = 'junior'
            break
        
    return seniority

def get_word_tokenize(text):
    # Tokenize a string of input text
    
    # Input
    # text: input text
    
    # Output
    # list of tokenized words 
       
    sentences = [s for s in nltk.sent_tokenize(text)]
    normalized_sentences = [s.lower() for s in sentences]
    return [w.lower() for sentence in normalized_sentences for w in nltk.word_tokenize(sentence)]  

def get_top_n_words(words, n, stopwords): 
    # Return the top n most frequent words from a tokenized list of words, using the input stopwords
    
    # Input
    # words: tokenized words
    # n: Top N words to return
    # stopwords: List of stopwords
    
    # Output
    # top_n_words: Top N most frequent words
      
    fdist = nltk.FreqDist(words)
    top_n_words = [w[0] for w in fdist.items() if w[0] not in stopwords][:n]    
    return top_n_words

def get_ngrams(n_gram, words, freq, n_best, stopwords):   
    # Get all Bigrams/Trigrams for input words
    
    # Input
    # n_gram: 2 (Bigram) or 3 (Trigram)
    # words: tokenized words
    # freq: Minimum number of occurances to count as an n-gram
    # n_best: Top N n-grams to return
    # stopwords: List of stopwords
    
    # Output
    # collocations: List of Top N n-gram tuples
    
    finder = None
    scorer = None
    if n_gram == 2:
        finder = nltk.BigramCollocationFinder.from_words(words)
        scorer = nltk.metrics.BigramAssocMeasures.jaccard
    elif n_gram == 3:
        finder = nltk.TrigramCollocationFinder.from_words(words)
        scorer = nltk.metrics.TrigramAssocMeasures.jaccard
    else:
        raise Exception('Only Bigrams and Trigrams are supported.')
    
    finder.apply_freq_filter(freq)  # Minimum number of occurances
    finder.apply_word_filter(lambda w: w in stopwords)
    collocations = finder.nbest(scorer, n_best)
    return collocations

def test_ds_words():
    
    # Reads the description and job position text from a file of job posts,
    # tokenizing the results to view top n words, bigrams, and trigrams. Results
    # are written to a text file. 
   
    job_data = read_pickle('Job Data.pkl')
    
    # Tokenize the job text
    summ_words = []
    desr_words = []    
    for _, row in job_data.iterrows():
        if row['position'] is not None:
            summ_words += get_word_tokenize(row['position'])    
        if row['description'] is not None:
            desr_words += get_word_tokenize(row['description']) 
      
    stopwords = nltk.corpus.stopwords.words('english')
    stopwords += [',', '.', ':', '(', ')', '-', ';', '&amp;', '!', '?','\'s']
    
    words_file = open("top_N_words.log","w")
      
    # Get the Top N words
    top_n_summ_words = get_top_n_words(summ_words, 10, stopwords)
    top_n_desr_words = get_top_n_words(desr_words, 50, stopwords) 
    
    print &gt;&gt; words_file, 'Top 10 Job Position Words\n'    
                     
    for top_word in top_n_summ_words:
        print &gt;&gt; words_file, top_word
          
    print &gt;&gt; words_file, '\n\n'
    
    print &gt;&gt; words_file, 'Top 50 Job Description Words\n'    
          
    for top_word in top_n_desr_words:
        print &gt;&gt; words_file, top_word
          
    min_hits = int(len(job_data) * 0.05)
    
    print &gt;&gt; words_file, 'Top 50 Job Description Bigrams\n'
          
    # Get the Bigrams
    big_2_words = get_ngrams(2, desr_words, min_hits, 50, stopwords)
      
    print &gt;&gt; words_file, '\n\n'
          
    for top_word in big_2_words:
        print &gt;&gt; words_file, ' '.join(top_word)
          
    print &gt;&gt; words_file, '\n\n'  
    
    print &gt;&gt; words_file, 'Top 50 Job Description Trigrams\n'          
               
    # Get the Trigrams
    big_3_words = get_ngrams(3, desr_words, min_hits, 50, stopwords)
      
    for top_word in big_3_words:
        print &gt;&gt; words_file, ' '.join(top_word)            
          
    words_file.close()
    
def normalize(s):
    # Normalizes + tokenizes input text + punctuation
    # from http://streamhacker.com/2011/10/31/fuzzy-string-matching-python/
    
    # Input
    # s: input text
    
    # Output
    # string of cleaned text
    
    words = tokenize.wordpunct_tokenize(s.lower().strip())
    return ' '.join([stemmer.stem(w) for w in words])
 
def fuzzy_match(s1, s2):
    # Calculates the edit distance between two normalized strings
    # from http://streamhacker.com/2011/10/31/fuzzy-string-matching-python/
    
    # Input
    # s1, s2: input text
    
    # Output
    # edit distance between the two strings
    
    return nltk.metrics.edit_distance(normalize(s1), normalize(s2))

def get_lnkin_code_name(company, field):
    # Gets the code, name values from an input dict
    
    # Input
    # company: dict
    # field: key in dict
    
    # Output
    # field_code: company[field]['code']
    # field_name: company[field]['name']
    
    field_code = None
    field_name = None    
    if field in company:
        field_code = company[field]['code']
        field_name = company[field]['name']
    return field_code, field_name

def get_year(dt):
    # Separates out the year from an input date
    
    # Input
    # dt: date
    
    # Output
    # year or None
    
    if dt is None:
        return None
    else:
        return dt.date().year

def get_month(dt):
    # Separates out the month from an input date
    
    # Input
    # dt: date
    
    # Output
    # month or None
        
    if dt is None:
        return None
    else:
        return dt.date().month
    
def get_word_count(text):
    # Returns number of words in an input text
    
    # Input
    # text: input text
    
    # Output
    # Number of words
    
    if text is not None:
        return len(text.split())
    else:
        return 0  
    
def get_char_count(text):
    # Returns number of characters in an input text
    
    # Input
    # text: input text
    
    # Output
    # Number of characters
    
    if text is not None:
        return sum(len(s) for s in text.split())
    else:
        return 0
    
def get_est_seniority_value(est_seniority):    
    # Converts the estimated seniority into an integer value
    
    # Input
    # est_seniority: text
    
    # Output
    # Integer
    
    return {
            'junior' : 1,
            'default' : 2,
            'senior' : 3
            }.get(est_seniority, -1)
            
def get_emply_count_value(employee_count_code):    
    # Converts the employee count code into an integer value,
    # which is the midpoint of the count range
    
    # Input
    # employee_count_code: char
    
    # Output
    # Integer
    
    return {
            'A' : 1,        # 1
            'B' : 6,        # 2 - 10
            'C' : 30,       # 11 - 50
            'D' : 125,      # 51 - 200
            'E' : 350,      # 201 - 500
            'F' : 750,      # 501 - 1,000
            'G' : 3000,     # 1,001 - 5,000
            'H' : 7500,     # 5,001 - 10,000
            'I' : 20000     # 10,000+
            }.get(employee_count_code, -1)   
            
def get_cmpny_type_value(company_type_code):    
    # Converts the company type code into an integer value
    
    # Input
    # company_type_code: char
    
    # Output
    # Integer
    
    return {
            'C' : 1,    # Public Company
            'D' : 2,    # Educational
            'N' : 3,    # Non-Profit
            'O' : 4,    # Self Owned
            'P' : 5,    # Privately Held
            'S' : 6     # Partnership
            }.get(company_type_code, -1) 

def round_to_thousands(x):  
    # Rounds normally to nearest thousand
    
    # Input
    # x: integer
    
    # Output
    # Rounded integer
    
    return int(math.floor(float(x) / 1000.0 + 0.5)) * 1000     
        
def update_company_data_from_linkedin():
    
    # Retrieves all of the company names from the job postings,
    # and queries LinkedIn for additional information
    
    # Define CONSUMER_KEY, CONSUMER_SECRET,  
    # USER_TOKEN, and USER_SECRET from the credentials 
    # provided in your LinkedIn application
    
    # Instantiate the developer authentication class
    
    authentication = linkedin.LinkedInDeveloperAuthentication(LINKEDIN_CONSUMER_KEY, LINKEDIN_CONSUMER_SECRET, 
                                                              LINKEDIN_OAUTH_USER_TOKEN, LINKEDIN_OAUTH_USER_SECRET, 
                                                              RETURN_URL, linkedin.PERMISSIONS.enums.values())
    
    # Pass it in to the app...
    
    application = linkedin.LinkedInApplication(authentication)    
    
    job_data = read_pickle('Job Data.pkl')
    company_list = np.unique(job_data.name.values.ravel())
        
    # Set dict of return values and inputs
    comp_sels = [{'companies': ['name', 'universal-name', 'description', 'company-type', 'industries', 'status', 'employee-count-range', 'specialties', 'website-url']}]
    comp_params = {'keywords' : None}
    
    # Data dictionaries - going to convert them into Pandas dataframes
    linkedin_companies = {}
    linkedin_industries = {}
    linkedin_specialities = {}
    
    # Loop through the unique set of companies
    for idx, comp_name in enumerate(company_list):
        comp_params['keywords'] = comp_name # Set company name as keyword       
        comp_vals = application.search_company(selectors = comp_sels, params = comp_params)
        
        if comp_vals['companies']['_total'] == 0:   # No results returned
            continue
        
        # Calculate the edit distance between the returned results and the input name
        dist_vals = []        
        for jdx, company in enumerate(comp_vals['companies']['values']):
            link_comp_name = company['name']
            name_dist = fuzzy_match(comp_name, link_comp_name)
            dist_vals.append([link_comp_name, name_dist, jdx])
            
        # Sort the values and choose the best one
        sort_dist_vals = sorted(dist_vals, key=lambda s: s[1])
        best_guess_company = comp_vals['companies']['values'][sort_dist_vals[0][2]]
        best_guess_name = sort_dist_vals[0][0]
        
        status_code, status_name = get_lnkin_code_name(best_guess_company, 'status')
        company_type_code, company_type_name = get_lnkin_code_name(best_guess_company, 'companyType')
        employee_count_code, employee_count_name = get_lnkin_code_name(best_guess_company, 'employeeCountRange')
        
        # Store company related data in a dictionary
        linkedin_company = {}
        linkedin_company['name'] = comp_name        
        linkedin_company['lnkn_name'] = best_guess_name        
        linkedin_company['lnkn_universal_name'] = best_guess_company.get('universalName')
        linkedin_company['lnkn_description'] = best_guess_company.get('description')
        linkedin_company['status_code'] = status_code
        linkedin_company['status_name'] = status_name
        linkedin_company['company_type_code'] = company_type_code
        linkedin_company['company_type_name'] = company_type_name
        linkedin_company['employee_count_code'] = employee_count_code
        linkedin_company['employee_count_name'] = employee_count_name
        linkedin_company['websiteUrl'] = best_guess_company.get('websiteUrl')                
        linkedin_companies[idx] = linkedin_company
                        
        # Store industry data in a separate dict
        if 'industries' in best_guess_company:
            if best_guess_company['industries']['_total'] &gt; 0:
                ind_start = len(linkedin_industries)
                for jdx, industry in enumerate(best_guess_company['industries']['values']):
                    linkedin_industry = {}
                    linkedin_industry['lnkn_name'] = best_guess_name
                    linkedin_industry['industry_type_code'] = industry['code']
                    linkedin_industry['industry_type_name'] = industry['name']
                    linkedin_industries[ind_start + jdx] = linkedin_industry
                
        # Store speciality data in a separate dict
        if 'specialties' in best_guess_company:
            if best_guess_company['specialties']['_total'] &gt; 0:
                spec_start = len(linkedin_specialities)
                for jdx, speciality in enumerate(best_guess_company['specialties']['values']):
                    linkedin_speciality = {}
                    linkedin_speciality['lnkn_name'] = best_guess_name
                    linkedin_speciality['speciality'] = speciality
                    linkedin_specialities[spec_start + jdx] = linkedin_speciality                
    
    # Convert to Pandas dataframes
    company_data = pd.DataFrame.from_dict(linkedin_companies, orient='index')
    industry_data = pd.DataFrame.from_dict(linkedin_industries, orient='index')
    speciality_data = pd.DataFrame.from_dict(linkedin_specialities, orient='index')
    
    # Pickle and write to spreadsheets
    company_data.to_pickle('LinkedIn Company Data.pkl')
    industry_data.to_pickle('LinkedIn Industry Data.pkl')
    speciality_data.to_pickle('LinkedIn Speciality Data.pkl')
    
    wrtr = pd.ExcelWriter('LinkedIn Data.xlsx')
    company_data.to_excel(wrtr, 'Companies')
    industry_data.to_excel(wrtr, 'Industries')
    speciality_data.to_excel(wrtr, 'Specialities')   
    wrtr.save()
    
    # Grab some simple statistics from the data generated and write it to a spreadsheet
    # for followup analysis
    
    employee_count = pd.DataFrame(company_data.groupby(['employee_count_name']).size())
    company_type = pd.DataFrame(company_data.groupby(['company_type_name']).size())    
    industry_count = pd.DataFrame(industry_data.groupby(['industry_type_name']).size())
    speciality_count = pd.DataFrame(speciality_data.groupby(['speciality']).size())
    
    wrtr = pd.ExcelWriter('LinkedIn Data Stats.xlsx')
    employee_count.to_excel(wrtr, 'Employee Count')
    company_type.to_excel(wrtr, 'Company Type')
    industry_count.to_excel(wrtr, 'Industry Count')
    speciality_count.to_excel(wrtr, 'Speciality Count')
    wrtr.save()
    
def prepare_and_merge_data():
    
    # Retrieves all dataframes and merges into a single dataframe
    # which is then pickled
    
    job_data = read_pickle('Job Data.pkl')
    company_data = read_pickle('LinkedIn Company Data.pkl')
    industry_data = read_pickle('LinkedIn Industry Data.pkl')
    speciality_data = read_pickle('LinkedIn Speciality Data.pkl')    
    
    # Add in derived data and fill in blank data
        
    job_data['post_year'] = job_data.date_posted.apply(get_year)    # Get date_posted year
    job_data['post_month'] = job_data.date_posted.apply(get_month)  # Get date_posted month
    job_data['desc_word_count'] = job_data.description.apply(get_word_count)    # Number of words in job description
    job_data['desc_char_count'] = job_data.description.apply(get_char_count)    # Number of characters in job description
    job_data['estimated_seniority_value'] = job_data.estimated_seniority.apply(get_est_seniority_value) # Convert estimated seniority to an integer
        
    company_data.loc[company_data.employee_count_code.isnull(), 'employee_count_code'] = 'D'    # '51-200'
    company_data.loc[company_data.company_type_code.isnull(), 'company_type_code'] = 'P'    # 'Privately Held'        
    company_data['employee_count_value'] = company_data.employee_count_code.apply(get_emply_count_value) # Convert employee count code to an integer
    company_data['company_type_value'] = company_data.company_type_code.apply(get_cmpny_type_value) # Convert company type code to an integer
    
    industry_data = pd.merge(industry_data, company_data[['lnkn_name']], how = 'right', on = 'lnkn_name')
    industry_data.loc[industry_data.industry_type_name.isnull(), 'industry_type_name'] = 'Unknown'
            
    # Converting the Industry and Speciality data into dataframes of frequencies
    # Only counting a subset of specialities as data science-y
    industry_group = industry_data[['lnkn_name', 'industry_type_name']].groupby(['lnkn_name', 'industry_type_name']).size().unstack('industry_type_name')        
    industry_group[industry_group.notnull()] = 1
    industry_group[industry_group.isnull()] = 0
        
    ds_specialities = ['Big Data', 'Analytics', 'Machine Learning', 'analytics', 'Data Science']
    ds_specialities.extend(['Big Data Analytics', 'Natural Language Processing', 'Predictive Analytics', 'Data Mining'])
    speciality_group = speciality_data[speciality_data.speciality.isin(ds_specialities)].groupby(['lnkn_name', 'speciality']).size().unstack('speciality')    
    speciality_group = pd.merge(speciality_group, company_data[['lnkn_name']], how = 'right', right_on = 'lnkn_name', left_index = True)   
    speciality_group.set_index('lnkn_name', inplace = True)
    speciality_group[speciality_group.notnull()] = 1
    speciality_group[speciality_group.isnull()] = 0
        
    # Merge the dataframes
    merge_data = pd.merge(job_data, company_data, on = 'name') 
    merge_data = pd.merge(merge_data, industry_group, left_on = 'lnkn_name', right_index = True)
    merge_data = pd.merge(merge_data, speciality_group, how = 'left', left_on = 'lnkn_name', right_index = True)
        
    merge_data.to_pickle('Clean Job Data.pkl')

def print_histogram(y, num_bins):   
    # Prints a histogram of input array with equally spaced bins
    
    # Input
    # y: array
    # num_bins: number of bins in histogram
    
    _, _, patches = py.hist(y, num_bins, histtype='stepfilled')
    py.setp(patches, 'facecolor', 'g', 'alpha', 0.75)
    py.show()

def get_train_test_sets(x, y, is_small_set):
    # Routine to consolidate the train/test set retrieval
    
    # Input
    # x, y: input arrays
    # is_small_set: whether the input sets are small (and thus we use a more even train/test split)
    
    # Output
    # split training, test sets 
    
    if is_small_set:
        return cv.train_test_split(x, y, train_size = 0.5, random_state = 0) 
    else:
        return cv.train_test_split(x, y, train_size = 0.8, random_state = 0)
    
def convert_to_salary_range(salary_value):
    # Groups salary amounts into ranges of 10k
    
    # Input
    # salary_value: salary amount
    
    # Output
    # Salary range value
    
    return int(math.floor(float(salary_value) / 10000.0 + 0.5))
    
def get_header_style():
    # Creates style object for xlwt spreadsheet
    # Font style - bold and underline
    
    header_font = xlwt.Font()
    header_font.bold = True
    header_font.underline = True    
    
    header_style = xlwt.XFStyle()
    header_style.font = header_font
        
    return header_style
    
def get_grid_search_values(model, grid_params, x_train, y_train, x_test, y_test, scoring_criteria = 'mean_squared_error'):  
    # Run a grid search on a model, and return the train / test score and MSE on the best result
    
    # Input
    # model: scikit-learn model
    # grid_params: dict of parameter space
    # x_train: independent variables training set
    # y_train: dependent variable training set
    # x_test: independent variables test set
    # y_test: dependent variable test set
    # scoring_criteria: model scoring criteria
    
    # Output
    # best_model: model that produced the best results
    # para_search.best_params_: best grid parameters
    # train_score: training score
    # test_score: test score
    # train_mse: training mse
    # test_mse: test mse
    
    para_search = grid_search.GridSearchCV(model, grid_params, scoring = scoring_criteria, cv = 5).fit(x_train, y_train)
    best_model = para_search.best_estimator_
    train_score = best_model.score(x_train, y_train)
    test_score = best_model.score(x_test, y_test)
    train_mse = metrics.mean_squared_error(best_model.predict(x_train), y_train)
    test_mse = metrics.mean_squared_error(best_model.predict(x_test), y_test)
    
    return best_model, para_search.best_params_, train_score, test_score, train_mse, test_mse 

def get_model_values(model, x_train, y_train, x_test, y_test):
    # Fit a model and return the score and mse
    
    # Input
    # model: scikit-learn model
    # x_train: independent variables training set
    # y_train: dependent variable training set
    # x_test: independent variables test set
    # y_test: dependent variable test set
    
    # Output
    # train_score: training score
    # test_score: test score
    # train_mse: training mse
    # test_mse: test mse        
    
    model.fit(x_train, y_train)
    
    train_score = model.score(x_train, y_train)
    test_score = model.score(x_test, y_test)
    train_mse = metrics.mean_squared_error(model.predict(x_train), y_train)
    test_mse = metrics.mean_squared_error(model.predict(x_test), y_test)
    
    return train_score, test_score, train_mse, test_mse

def get_best_k_model(model, max_k, x, y):
    # Fit a model using a range of best-k values, 
    # returning the model that produces the best test score
    
    # Input
    # model: scikit-learn model
    # max_k: maximum k-value to iterate to (inclusive)
    # x: independent variables
    # y: dependent variable
    
    # Output
    # best_k: Number of dependent variables using to produce output
    # train_score: training score
    # test_score: test score
    # train_mse: training mse
    # test_mse: test mse       
    
    test_scores = []
    k_vals = []    
    
    k_limit = min(max_k, len(x.columns))
    for k_val in range(1, k_limit + 1):
        best_x = fs.SelectKBest(fs.chi2, k = k_val).fit_transform(x, y)
        x_train, x_test, y_train, y_test = cv.train_test_split(best_x, y, test_size = 0.2, random_state = 0)
        test_scores.append(model.fit(x_train, y_train).score(x_test, y_test))
        k_vals.append(k_val)

    best_k = k_vals[np.argmax(test_scores)]
    best_x = fs.SelectKBest(fs.chi2, k = best_k).fit_transform(x, y)
    x_train, x_test, y_train, y_test = cv.train_test_split(best_x, y, test_size = 0.2, random_state = 0)
       
    train_score, test_score, train_mse, test_mse = get_model_values(model, x_train, y_train, x_test, y_test)
    
    return best_k, train_score, test_score, train_mse, test_mse

def get_clean_column_names(job_data, drop_cols):    
    # Configure the final set of columns and remove any columns with flat values
    
    # Input
    # job_data: Pandas dataframe
    # drop_cols: columns we're excluding
    
    # Output
    # data_cols: non-excluded columns that have non-flat values
    
    data_cols = []
    for col in job_data.columns.values.tolist():
        if col in drop_cols:
            continue
        col_vals = job_data[col].values.ravel()
        if np.min(col_vals) != np.max(col_vals):
            data_cols.append(col)
            
    return data_cols

def normalize_and_apply_pca(job_data, pca):
    # Normalize a series of job data and then apply PCA
    
    # Input
    # job_data: Pandas dataframe
    # pca: scikit-learn PCA object
    
    # Output
    # transformed normalized data
    
    norm_data = (job_data - job_data.mean()) / job_data.std()
    return pca.fit_transform(norm_data)

def write_to_spreadsheet(model_name, dataset_name, train_score, test_score, train_mse, test_mse, best_k, best_params, sh, row):
    # Write a set of data to an xlwt spreadsheet
    
    # Input
    # model_name: Name of model
    # dataset_name: Name of dataset
    # train_score: Training score
    # test_score: Test score
    # train_mse: Training MSE
    # test_mse: Test MSE
    # best_k: Best-K value, if we used Best-K for modeling
    # best_params: Best Grid parameters, if we used grid modeling
    # sh: xlwt spreadsheet
    # row: row number
    
    # Output
    # updates the row number

    sh.write(row, 0, model_name)
    sh.write(row, 1, dataset_name)
    sh.write(row, 2, train_score)
    sh.write(row, 3, test_score)
    sh.write(row, 4, train_mse)
    sh.write(row, 5, test_mse)
    if best_k is not None:
        sh.write(row, 6, best_k)
    if best_params is not None:
        sh.write(row, 7, str(best_params))
    
    return row + 1
    
def run_model():    
    # Run the models against the data and write the results to a spreadsheet
    
    # Retrieve the data that we processed in prepare_and_merge_data()
    clean_job_data = read_pickle('Clean Job Data.pkl')
    clean_job_data = clean_job_data[clean_job_data.estimated_salary &gt; 0.0]
        
    # These are the columns that we don't need
    drop_cols = ['company_id', 'date_posted', 'description', 'estimated_salary', 'expiration_date']
    drop_cols.extend(['job_id', 'pay_rate', 'position', 'skill_count', 'source_uri', 'estimated_seniority'])
    drop_cols.extend(['name', 'company_industry', 'company_type', 'is_public', 'number_of_employees', 'status_name'])
    drop_cols.extend(['status_code', 'lnkn_description', 'websiteUrl', 'employee_count_code', 'lnkn_universal_name'])
    drop_cols.extend(['company_type_name', 'lnkn_name', 'employee_count_name', 'company_type_code', 'clean_pay_rate_annualized'])
    
    # Use records that have the pay rate provided in the job post - this is a small set
    pay_rate_data = clean_job_data[clean_job_data.clean_pay_rate_annualized.notnull()]
    pay_cols = get_clean_column_names(pay_rate_data, drop_cols)
    print 'Number of Clean Pay Rate Records: {}'.format(len(pay_rate_data))                 
    x1 = pay_rate_data[pay_cols].astype(int)
    y1 = pay_rate_data.clean_pay_rate_annualized
    _, _, y1_train, y1_test = get_train_test_sets(x1, y1, True)
    print '{} Training Records / {} Testing Records'.format(y1_train.size, y1_test.size)
    
    # Use records that have an estimated salary, which we will round to nearest 1k
    print 'Number of Estimated Salary Records: {}'.format(len(clean_job_data))
    est_sal_cols = get_clean_column_names(clean_job_data, drop_cols)
    x2 = clean_job_data[est_sal_cols].astype(int)
    y2 = clean_job_data.estimated_salary.apply(round_to_thousands) 
    _, _, y2_train, y2_test = get_train_test_sets(x2, y2, False)
    print '{} Training Records / {} Testing Records'.format(y2_train.size, y2_test.size)
    
    # Different approach - groups salaries in amounts of 10k and see if we can get better results
    y3 = pay_rate_data.clean_pay_rate_annualized.apply(convert_to_salary_range) # Convert Pay Rate to a range
    y4 = clean_job_data.estimated_salary.apply(convert_to_salary_range) # Convert Est Salary to a range
    
    # Transform the independent variables using PCA to see if that helps on some of the models
    pca = PCA().set_params(n_components = 0.9)     
    x3 = normalize_and_apply_pca(x1, pca)
    x4 = normalize_and_apply_pca(x2, pca)
                
    results_book = xlwt.Workbook()    
    head_style = get_header_style()
    
    pyrt_sh = results_book.add_sheet('Pay Rate')
    pyrt_sh.write(0, 0, "Model Name", head_style)
    pyrt_sh.write(0, 1, "Dataset", head_style)
    pyrt_sh.write(0, 2, "Training Score", head_style)
    pyrt_sh.write(0, 3, "Testing Score", head_style)
    pyrt_sh.write(0, 4, "Training MSE", head_style)
    pyrt_sh.write(0, 5, "Testing MSE", head_style)
    pyrt_sh.write(0, 6, "Best K", head_style)
    pyrt_sh.write(0, 7, "Best Parameters", head_style)
    
    estsal_sh = results_book.add_sheet('Est Salary')
    estsal_sh.write(0, 0, "Model Name", head_style)
    estsal_sh.write(0, 1, "Dataset", head_style)
    estsal_sh.write(0, 2, "Training Score", head_style)
    estsal_sh.write(0, 3, "Testing Score", head_style)
    estsal_sh.write(0, 4, "Training MSE", head_style)
    estsal_sh.write(0, 5, "Testing MSE", head_style)
    estsal_sh.write(0, 6, "Best K", head_style)
    estsal_sh.write(0, 7, "Best Parameters", head_style)    
    
    # Do an initial test using linear models with different shapes for the dependent variable
    linear_datasets = [("Pay Rate", x1, y1, True),
                ("Log Pay Rate", x1, np.log(y1), True),
                ("Sqrt Pay Rate", x1, np.sqrt(y1), True),
                ("Est Salary", x2, y2, False),
                ("Log Est Salary", x2, np.log(y2), False),
                ("Sqrt Est Salary", x2, np.sqrt(y2), False),
                ("Pay Rate Range", x1, y3, True),
                ("Log Pay Rate Range", x1, np.log(y3), True),
                ("Sqrt Pay Rate Range", x1, np.sqrt(y3), True),
                ("Est Salary Range", x2, y4, False),
                ("Log Est Salary Range", x2, np.log(y4), False),
                ("Sqrt Est Salary Range", x2, np.sqrt(y4), False)                
                ]
         
    linear_models = [("OLS", linear_model.LinearRegression()),
              ("Ridge", linear_model.RidgeCV(normalize = True, fit_intercept = False, scoring = 'mean_squared_error', cv = 5)),
              ("Lasso", linear_model.LassoCV(normalize = True, fit_intercept = False, cv = 5))]
    
    prow = 1
    erow = 1
    for data in linear_datasets:
        x_train, x_test, y_train, y_test = get_train_test_sets(data[1], data[2], data[3])
          
        for model in linear_models:            
            train_score, test_score, train_mse, test_mse = get_model_values(model[1], x_train, y_train, x_test, y_test)
            
            if data[3] == True:
                prow = write_to_spreadsheet(model[0], data[0], train_score, test_score, train_mse, test_mse, None, None, pyrt_sh, prow)
            else:
                erow = write_to_spreadsheet(model[0], data[0], train_score, test_score, train_mse, test_mse, None, None, estsal_sh, erow)
       
    # Test on a different set of models, where we're applying PCA to reduce the number of features        
    datasets = [("Pay Rate", x1, y1, True),
                ("PCA Pay Rate", x3, y1, True),
                ("Pay Rate Range", x1, y3, True),
                ("PCA Pay Rate Range", x3, y3, True),
                ("Est Salary", x2, y2, False),
                ("PCA Est Salary", x4, y2, False),
                ("Est Salary Range", x2, y4, False),
                ("PCA Est Salary Range", x4, y4, False)             
                ]
    
    models = [("KNN", neighbors.KNeighborsClassifier(), {'n_neighbors' : np.arange(3, 9), 'weights' : ['uniform', 'distance'], 'p' : [1, 2]}),
              ("Decision Tree", tree.DecisionTreeClassifier(), {'criterion' : ['gini', 'entropy'], 'max_features' : [None, 'auto', 'log2']}),
              ("Random Forest", ensemble.RandomForestClassifier(), {'criterion': ['gini', 'entropy'], 'max_features' : [None, 'auto', 'log2'], 'n_estimators': np.arange(10, 110, 10)})
              ]
    
    for data in datasets:         
        x_train, x_test, y_train, y_test = get_train_test_sets(data[1], data[2], data[3])    
     
        for model in models:
            _, best_params, train_score, test_score, train_mse, test_mse = get_grid_search_values(model[1], model[2], x_train, y_train, x_test, y_test, 'accuracy')                 
            
            if data[3] == True:
                prow = write_to_spreadsheet(model[0], data[0], train_score, test_score, train_mse, test_mse, None, best_params, pyrt_sh, prow)
            else:
                erow = write_to_spreadsheet(model[0], data[0], train_score, test_score, train_mse, test_mse, None, best_params, estsal_sh, erow)            

    # Use the best K on LDA - had collinearity issues with full feature set              
    datasets = [("Pay Rate Range Best K", x1, y3.values.ravel(), True),
                ("Est Salary Range Best K", x2, y4.values.ravel(), False)              
                ]
    
    models = [("LDA", lda.LDA())]
    
    for data in datasets:         
        for model in models: 
            best_k, train_score, test_score, train_mse, test_mse = get_best_k_model(model[1], 20, data[1], data[2])                            
            
            if data[3] == True:
                prow = write_to_spreadsheet(model[0], data[0], train_score, test_score, train_mse, test_mse, best_k, None, pyrt_sh, prow)
            else:
                erow = write_to_spreadsheet(model[0], data[0], train_score, test_score, train_mse, test_mse, best_k, None, estsal_sh, erow)            
                        
    results_book.save("Model Results.xls")         
    
run_model()
