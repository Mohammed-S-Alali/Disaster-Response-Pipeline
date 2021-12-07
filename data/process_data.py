import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    This function loads messages and categories datasets by filepath

    Parameters
    ----------
    messages_filepath:str, path object or file-like object for messages dataset\n
    Any valid string path is acceptable. The string could be a URL.

    categories_filepath:str, path object or file-like object for categorie dataset\n
    Any valid string path is acceptable. The string could be a URL.

    Returns
    -------
    DataFrame
        A merged two csv files is returned as two-dimensional data structure with labeled axes.
    """
    
    # load the csv file of messages dataset
    messages = pd.read_csv(messages_filepath)
    
    # load the csv file of categories dataset
    categories = pd.read_csv(categories_filepath).to_excel
    
    # merge two datasets by Id attribute
    merged_df = messages.merge(categories, how='inner', on='id')
    
    return merged_df


def clean_data(df):
    """
    This function cleans the dataframe and removes the incorrect rows

    Parameters
    ----------
    df:DataFrame\n
    Any valid two-dimensional data structure with labeled axes is acceptable.

    Returns
    -------
    DataFrame
        A cleaned dataframe is returned as two-dimensional data structure with labeled axes.
    """

    # Split `categories` into separate category columns
    categories = df['categories'].str.split(';', expand=True)
    
    # select first row
    row = categories.loc[0]

    # use this row to extract a list of new column names for categories
    new_columns = [r[:-2] for r in row]

    # rename the columns of `categories`
    categories.columns = new_columns

    # convert category values to just numbers 0 or 1
    for column in categories:

        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1]

        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    
    # drop the original categories column from `df`
    df.drop('categories', axis = 1, inplace = True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    # the related column contains some values of 2, to make this binary, map the 2s to 1s
    df['related'] = df['related'].map({0:0,1:1,2:1})

    # drop the duplicates
    df.drop_duplicates(inplace=True)

    return df


def save_data(df, database_filename):
    """
    This function saves the dataframe in sql format with user-defined name.

    Parameters
    ----------
    df:DataFrame\n
    Any valid two-dimensional data structure with labeled axes is acceptable.

    database_filename:str\n
    Name of database file which will contain DataFrame.

    Returns
    -------
    No returns    
    """

    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('Messages', engine, index=False, if_exists='replace')  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()