import pandas as pd
import numpy as np

def interest_generator(df_favorites=None,df_bookmarks=None,df_ratings=None):
    '''
    computes interests scores from tables favorites.csv, bookmarks.csv and ratings.csv;

    df_bookmarks: pandas dataframe of shape (nrows_book,3)
    df_favorites: pandas dataframe of shape (nrows_fav,3)
    df_ratings: pandas dataframe of shape (nrows_rat,4)

    return: dataframe of shape (nrows_book,3)
    '''

    m_ui=1

    n_ui=pd.merge(df_bookmarks,df_ratings,how='left',on=['id_profile','id_asset'])
    n_ui['n_ui'] = n_ui['score'].fillna(0)
    n_ui.drop(['time_x','time_y','score'],axis=1,inplace=True)

    f_ui=pd.merge(df_bookmarks,df_favorites,how='left',on=['id_profile','id_asset'])
    f_ui.drop(['time'],axis=1,inplace=True)
    f_ui['f_ui'] = np.where(f_ui['added_date'].isna(), 0, 5)
    f_ui.drop(['added_date'],axis=1,inplace=True)

    df_interest=pd.DataFrame({
        'id_profile':df_bookmarks['id_profile'],
        'id_asset':df_bookmarks['id_asset'],
        'interest':m_ui+n_ui['n_ui']+f_ui['f_ui']
        })

    return df_interest

def split_dataset(df_interest):
  '''
  
  '''
  idx_train = np.load('recommendation/bookmarks_idx_train.npy')
  idx_test = np.load('recommendation/bookmarks_idx_test.npy')
  
  train_interest=df_interest.loc[idx_train]
  test_interest=df_interest.loc[idx_test]

  return train_interest,test_interest